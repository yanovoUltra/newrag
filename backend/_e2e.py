"""端到端验证：生产路由上线后指标题/分析题行为。"""
import json
import requests

BASE = "http://localhost:8000/api/v1/chat"

def ask(q: str):
    print(f"\n===== {q}")
    r = requests.post(BASE, json={"question": q, "session_id": "", "org_id": "default", "user_visibility": "public"}, stream=True, timeout=120)
    meta, tokens, done = None, "", None
    cur = None
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("event:"):
            cur = line[6:].strip()
        elif line.startswith("data:"):
            ev = json.loads(line[5:])
            if cur == "meta":
                meta = ev
            elif cur == "token":
                tokens += ev.get("delta", "")
            elif cur == "done":
                done = ev
    print(f"  meta: intent={meta.get('intent')} complexity={meta.get('complexity')} "
          f"query_type={meta.get('query_type')} needs_hyde={meta.get('needs_hyde')} top_k={meta.get('top_k')}")
    print(f"  回答: {tokens[:120]}")
    if done:
        print(f"  usage: {json.dumps(done.get('usage', {}), ensure_ascii=False)[:120]}")

ask("浦发银行2024年不良贷款率是多少？")
ask("根据招商银行年报，分析本集团净利息收益率未来趋势如何？")
ask("为什么公司2024年经营活动现金流量净额相比上年显著增加？")
