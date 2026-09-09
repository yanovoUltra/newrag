"""Coverage-aware 4-bit QLoRA training for Qwen3-Reranker-8B.

Each step samples several positive aspects and hard negatives for one query.  The loss
combines yes/no classification with a smooth worst-positive-vs-best-negative margin,
so improving only the easiest evidence aspect is insufficient.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

INSTRUCTION_FALLBACK = (
    "Given a financial-report question, rank passages that jointly provide complete, "
    "company-correct, period-correct, and citable evidence."
)
PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    '<|im_start|>user\n'
)
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def format_pair(instruction: str, query: str, document: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-Reranker-8B")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--positives-per-step", type=int, default=2)
    parser.add_argument("--negatives-per-step", type=int, default=2)
    parser.add_argument(
        "--cover-all-positives",
        action="store_true",
        help="Repeat multi-aspect queries so every positive is sampled once per epoch",
    )
    parser.add_argument("--coverage-weight", type=float, default=0.5)
    parser.add_argument("--listwise-weight", type=float, default=0.5)
    parser.add_argument("--pairwise-weight", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means all epochs")
    parser.add_argument("--seed", type=int, default=20260817)
    return parser.parse_args()


def _sample(items: list[dict], count: int, offset: int) -> list[dict]:
    if not items:
        return []
    return [items[(offset + i) % len(items)] for i in range(min(count, len(items)))]


def main() -> int:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if os.name != "nt":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        get_linear_schedule_with_warmup,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3-Reranker-8B QLoRA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This training configuration requires BF16-capable CUDA hardware")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train_rows = load_jsonl(args.train)
    dev_rows = load_jsonl(args.dev)
    if not train_rows or not dev_rows:
        raise ValueError("Both train and dev datasets must be non-empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        # Native Windows has only 16 GiB system RAM here.  Quantizing an 8B
        # safetensors shard can otherwise overlap with the next CPU state dict
        # and force the desktop into paging before the tensors reach the GPU.
        offload_state_dict=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.print_trainable_parameters()

    prefix_tokens = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(SUFFIX, add_special_tokens=False)
    token_no = tokenizer.convert_tokens_to_ids("no")
    token_yes = tokenizer.convert_tokens_to_ids("yes")
    if token_no == tokenizer.unk_token_id or token_yes == tokenizer.unk_token_id:
        raise ValueError("Tokenizer does not expose single-token yes/no labels")
    body_budget = args.max_length - len(prefix_tokens) - len(suffix_tokens)
    if body_budget < 128:
        raise ValueError("max_length is too small after the official reranker prompt tokens")

    def encode_pairs(row: dict, positives: list[dict], negatives: list[dict]):
        bodies = [
            format_pair(row.get("instruction") or INSTRUCTION_FALLBACK, row["query"], item["text"])
            for item in positives + negatives
        ]
        encoded = tokenizer(
            bodies,
            padding=False,
            truncation=True,
            max_length=body_budget,
            add_special_tokens=False,
        )["input_ids"]
        sequences = [prefix_tokens + ids + suffix_tokens for ids in encoded]
        batch = tokenizer.pad(
            {"input_ids": sequences}, padding=True, return_tensors="pt"
        ).to(model.device)
        labels = torch.tensor(
            [1] * len(positives) + [0] * len(negatives), device=model.device, dtype=torch.long
        )
        return batch, labels

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    samples_per_epoch = sum(
        math.ceil(len(row["positives"]) / args.positives_per_step)
        if args.cover_all_positives
        else 1
        for row in train_rows
    )
    updates_per_epoch = math.ceil(samples_per_epoch / args.gradient_accumulation)
    total_updates = args.max_steps or updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, int(total_updates * 0.05)), num_training_steps=total_updates
    )

    def compute_loss(row: dict, offset: int):
        positives = _sample(row["positives"], args.positives_per_step, offset)
        negatives = _sample(row["negatives"], args.negatives_per_step, offset)
        batch, labels = encode_pairs(row, positives, negatives)
        # Qwen3 supports returning only the final-token logits.  Keeping logits for
        # every prompt position would materialize a [batch, seq, vocab] tensor and
        # wastes several GiB on a 151k-token vocabulary.
        logits = model(**batch, logits_to_keep=1).logits[:, -1, :][
            :, [token_no, token_yes]
        ].float()
        classification = F.cross_entropy(logits, labels)
        relevance = logits[:, 1] - logits[:, 0]
        pos_scores = relevance[: len(positives)]
        neg_scores = relevance[len(positives) :]

        # Graded listwise supervision rewards putting every evidence aspect ahead of
        # production-mined negatives, while preserving stronger labels (grade 3 > 2 > 1).
        grades = torch.tensor(
            [max(float(item.get("grade") or 1.0), 0.1) for item in positives],
            device=model.device,
        )
        target = torch.cat([grades / grades.sum(), torch.zeros_like(neg_scores)])
        listwise = -(target * F.log_softmax(relevance / args.temperature, dim=0)).sum()

        # Pair every sampled aspect with every hard negative instead of learning only
        # independent yes/no labels.  Grade weights do not change the binary target;
        # they control how strongly a more essential evidence group affects ranking.
        pair_margins = pos_scores[:, None] - neg_scores[None, :]
        grade_weights = grades / grades.mean()
        pairwise = (
            F.softplus(args.margin - pair_margins) * grade_weights[:, None]
        ).mean()

        # Coverage regularisation is intentionally based on the weakest aspect.  A
        # result is not complete when only one of several required evidence groups wins.
        by_aspect: dict[str, list] = {}
        for index, item in enumerate(positives):
            aspect_id = str(item.get("aspect_id") or f"aspect-{index}")
            by_aspect.setdefault(aspect_id, []).append(pos_scores[index])
        aspect_scores = torch.stack(
            [torch.stack(scores).min() for scores in by_aspect.values()]
        )
        smooth_min_pos = -args.temperature * torch.logsumexp(
            -aspect_scores / args.temperature, dim=0
        )
        smooth_max_neg = args.temperature * torch.logsumexp(neg_scores / args.temperature, dim=0)
        coverage = F.softplus(smooth_max_neg - smooth_min_pos + args.margin)
        loss = (
            classification
            + args.listwise_weight * listwise
            + args.pairwise_weight * pairwise
            + args.coverage_weight * coverage
        )
        return (
            loss,
            classification.detach(),
            listwise.detach(),
            pairwise.detach(),
            coverage.detach(),
        )

    @torch.no_grad()
    def evaluate(limit: int = 40) -> dict:
        model.eval()
        pair_correct = pair_total = group_hit = 0
        losses = []
        for index, row in enumerate(dev_rows[:limit]):
            positives = _sample(row["positives"], args.positives_per_step, index)
            negatives = _sample(row["negatives"], args.negatives_per_step, index)
            batch, labels = encode_pairs(row, positives, negatives)
            logits = model(**batch, logits_to_keep=1).logits[:, -1, :][
                :, [token_no, token_yes]
            ].float()
            losses.append(float(F.cross_entropy(logits, labels)))
            predicted = logits.argmax(dim=-1)
            pair_correct += int((predicted == labels).sum())
            pair_total += labels.numel()
            score = logits[:, 1] - logits[:, 0]
            group_hit += int(score[: len(positives)].min() > score[len(positives) :].max())
        model.train()
        return {
            "loss": sum(losses) / max(len(losses), 1),
            "pair_accuracy": pair_correct / max(pair_total, 1),
            "sampled_all_hit": group_hit / max(min(len(dev_rows), limit), 1),
            "groups": min(len(dev_rows), limit),
        }

    def save_checkpoint(update_step: int, state: dict) -> Path:
        path = args.output_dir / f"checkpoint-{update_step:05d}"
        model.save_pretrained(path, safe_serialization=True)
        tokenizer.save_pretrained(path)
        (path / "trainer_state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    baseline = evaluate()
    print("baseline", json.dumps(baseline, ensure_ascii=False), flush=True)
    history = [{"step": 0, "eval": baseline}]
    optimizer.zero_grad(set_to_none=True)
    update_step = 0
    micro_step = 0
    started = time.time()
    stop = False
    for epoch in range(args.epochs):
        order: list[tuple[int, int | None]] = []
        for row_index, row in enumerate(train_rows):
            repeats = (
                math.ceil(len(row["positives"]) / args.positives_per_step)
                if args.cover_all_positives
                else 1
            )
            for repeat in range(repeats):
                offset = repeat * args.positives_per_step if args.cover_all_positives else None
                order.append((row_index, offset))
        random.Random(args.seed + epoch).shuffle(order)
        model.train()
        for row_index, sample_offset in order:
            row = train_rows[row_index]
            loss, classification, listwise, pairwise, coverage = compute_loss(
                row, micro_step if sample_offset is None else sample_offset
            )
            (loss / args.gradient_accumulation).backward()
            micro_step += 1
            if micro_step % args.gradient_accumulation:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update_step += 1
            print(json.dumps({
                "step": update_step,
                "epoch": epoch,
                "loss": round(float(loss.detach()), 6),
                "classification": round(float(classification), 6),
                "listwise": round(float(listwise), 6),
                "pairwise": round(float(pairwise), 6),
                "coverage": round(float(coverage), 6),
                "lr": scheduler.get_last_lr()[0],
                "vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                "elapsed_s": round(time.time() - started, 1),
            }), flush=True)
            if update_step % args.eval_steps == 0:
                metrics = evaluate()
                history.append({"step": update_step, "eval": metrics})
                print("eval", json.dumps(metrics, ensure_ascii=False), flush=True)
            if update_step % args.save_steps == 0:
                save_checkpoint(
                    update_step,
                    {"step": update_step, "history": history, "args": serializable_args},
                )
            if args.max_steps and update_step >= args.max_steps:
                stop = True
                break
        if stop:
            break

    if micro_step % args.gradient_accumulation and not stop:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        update_step += 1
    final_eval = evaluate(limit=len(dev_rows))
    history.append({"step": update_step, "eval": final_eval})
    final_path = args.output_dir / "final-adapter"
    model.save_pretrained(final_path, safe_serialization=True)
    tokenizer.save_pretrained(final_path)
    (args.output_dir / "run_summary.json").write_text(
        json.dumps({
            "model": args.model,
            "steps": update_step,
            "baseline": baseline,
            "final": final_eval,
            "history": history,
            "args": serializable_args,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("final", json.dumps(final_eval, ensure_ascii=False), flush=True)
    print(f"adapter={final_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
