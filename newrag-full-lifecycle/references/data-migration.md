# Consistent data migration

## Contents

1. Define the consistency set
2. Inventory and filter
3. Snapshot and transfer
4. Restore order
5. Verify and release
6. Failure patterns

## Define the consistency set

Migrate these components together:

- SQLite registry and structured financial fields;
- original uploads;
- parsed pages, sections, chunks, checkpoints, and IDF artifacts;
- Qdrant collection configuration, vectors, sparse vectors, and payloads;
- evidence identifiers linking fields or answers to chunks.

Choose a cutoff and quiesce API/worker writers before snapshotting. A file-only or vector-only copy is not a complete migration.

## Inventory and filter

Record counts grouped by organization, visibility, status, document ID, and data type. Identify synthetic documents and test vectors from explicit evidence, not filename guesses alone. Confirm every formal document ID expected in Qdrant and every vector document ID expected in SQLite.

Use SQLite `PRAGMA integrity_check`. Calculate file counts, byte totals, and cryptographic hashes. Export only the intended organization when production must exclude test tenants.

## Snapshot and transfer

Use SQLite's online backup mechanism or stop writers and checkpoint WAL before copying. Create a Qdrant collection snapshot through its snapshot API. Archive uploads and pipeline artifacts while preserving paths and permissions.

Create destination pre-migration backups before replacing any state. Transfer to a staging directory, verify hashes at the destination, and only then restore.

## Restore order

1. Stop API and worker; leave required datastores available for snapshot recovery.
2. Restore application files and pipeline artifacts.
3. Remove or isolate stale SQLite `registry.db-wal` and `registry.db-shm` while writers are stopped.
4. Place the verified SQLite main file atomically and set the application UID/GID and permissions.
5. Recover the Qdrant snapshot using its API from a network context that can reach Qdrant.
6. Remove test records only by exact IDs or a verified metadata filter.
7. Start API and worker and wait for health.

Do not overwrite only `registry.db` while old WAL/SHM files remain: SQLite can replay old state and make the restored database appear empty.

## Verify and release

Perform all checks before declaring success:

- SQLite integrity is `ok`.
- formal document count/status and financial-field count match the manifest.
- upload and pipeline file counts and sizes match.
- Qdrant collection and filtered organization counts match.
- `DB document IDs - vector document IDs` is empty.
- `vector document IDs - DB document IDs` is empty.
- every non-empty `source_chunk_id` resolves to a Qdrant point.
- document API shows the expected records.
- representative RAG/competitor analysis returns evidence.
- worker pings, health is green, and recent logs are clean.

Run `scripts/verify_newrag_consistency.py` for a read-only subset of these checks. Save its JSON output in the migration evidence.

## Failure patterns

- **Online library is empty after deployment**: the new Compose project likely mounted new named volumes; inspect actual mounts and project-prefixed volume names.
- **Copied SQLite becomes empty/small**: stale WAL/SHM was replayed; stop writers, remove only those two exact sidecars, restore again.
- **Qdrant snapshot upload is unreliable through the public gateway**: transfer the snapshot over SSH and call Qdrant over the internal Docker network.
- **Point counts differ slightly**: group by organization and document ID; test points often explain the delta. Never delete the full collection to correct a small difference.
- **Fields exist but citations fail**: validate `source_chunk_id` and migrated pipeline artifacts, not only `doc_id`.
