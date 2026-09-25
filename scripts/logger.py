"""Hash-chained, append-only chain-of-custody log (JSON Lines).

Each record carries the SHA-256 of the previous record (`prev_hash`) and its own hash over the canonical JSON of
every other field. Editing, deleting, inserting or reordering any historical record breaks the chain from that
point on, and `verify()` reports the first record that no longer checks out.

Limitation: a hash chain alone cannot detect truncation of the newest records (deleting the tail leaves a valid,
shorter chain). A deployment would periodically anchor the head hash somewhere the log writer cannot modify.
"""

import argparse
import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

# REFUSED is the system working correctly (fail closed: it declined to file rather than guess); FAILED is an execution
# failure. They are different outcomes and must never be conflated.
ACTIONS = ("SUBMITTED", "REGISTERED", "PICKED", "PLACED", "VERIFIED", "FAILED", "REFUSED")
FIELDS = ("event_id", "timestamp", "actor", "action", "item_id", "case_id", "object_class", "slot_id", "detail",
          "prev_hash")
ROBOT_ACTOR = "ROBOT:arm-01"
SYSTEM_ACTOR = "SYSTEM"


def officer(badge):
    return f"OFFICER:{badge}"


def _valid_actor(actor):
    return actor in (ROBOT_ACTOR, SYSTEM_ACTOR) or (actor.startswith("OFFICER:") and len(actor) > len("OFFICER:"))


def canonical(record):
    body = {k: record[k] for k in FIELDS}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_hash(record):
    return hashlib.sha256(canonical(record).encode("utf-8")).hexdigest()


def format_event(ev):
    slot = ev["slot_id"] or "-"
    return (f"{ev['timestamp']}  {ev['actor']:<16} {ev['action']:<10} {ev['item_id']:<16} {ev['case_id']:<14} "
            f"{ev['object_class']:<9} {slot:<7} #{ev['hash'][:10]}  {ev['detail']}")


class CustodyLog:
    def __init__(self, path, echo=True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self._events = []
        if self.path.exists():
            bad = verify_file(self.path)
            if bad is not None:
                raise ValueError(f"{self.path} fails verification at record {bad}; refusing to append to a broken chain")
            self._events = self._read()

    def _read(self):
        with self.path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def append(self, actor, action, item_id, case_id, object_class, slot_id=None, detail=""):
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}; expected one of {ACTIONS}")
        if not _valid_actor(actor):
            raise ValueError(f"invalid actor {actor!r}")
        record = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"),
            "actor": actor,
            "action": action,
            "item_id": item_id,
            "case_id": case_id,
            "object_class": object_class,
            "slot_id": slot_id,
            "detail": detail,
            "prev_hash": self._events[-1]["hash"] if self._events else None,
        }
        record["hash"] = compute_hash(record)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._events.append(record)
        if self.echo:
            print(format_event(record))
        return record

    def verify(self):
        """Re-read the file and recompute the whole chain. Returns the index of the first bad record, or None."""
        return verify_file(self.path)

    def tail(self, n):
        return list(self._events[-n:])

    def __len__(self):
        return len(self._events)


def verify_file(path):
    """Index of the first record in the file that fails the chain (or does not parse), or None if intact."""
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                return len(records)
    return verify_records(records)


def verify_records(records):
    prev = None
    for i, rec in enumerate(records):
        if not isinstance(rec, dict) or any(k not in rec for k in FIELDS + ("hash",)):
            return i
        if rec["prev_hash"] != prev or compute_hash(rec) != rec["hash"]:
            return i
        prev = rec["hash"]
    return None


def _demo(out_dir):
    """Write a short, realistic custody log, verify it, then verify a tampered copy."""
    path = Path(out_dir) / "custody_log_demo.jsonl"
    path.unlink(missing_ok=True)
    log = CustodyLog(path)
    items = [("EV-2026-000101", "CASE-26-0412", "box", "slot_0"),
             ("EV-2026-000102", "CASE-26-0412", "bag", "slot_1")]
    for item_id, case_id, cls, slot in items:
        log.append(officer("4471"), "SUBMITTED", item_id, case_id, cls, detail="presented at intake counter")
        log.append(SYSTEM_ACTOR, "REGISTERED", item_id, case_id, cls, slot, detail=f"allocated {slot}")
        log.append(ROBOT_ACTOR, "PICKED", item_id, case_id, cls, slot, detail="grasp verified")
        log.append(ROBOT_ACTOR, "PLACED", item_id, case_id, cls, slot, detail="released in slot")
        log.append(SYSTEM_ACTOR, "VERIFIED", item_id, case_id, cls, slot, detail="at rest inside slot volume")
    print(f"clean log: verify() -> {log.verify()}")

    tampered = path.with_name("custody_log_demo_tampered.jsonl")
    shutil.copy(path, tampered)
    lines = tampered.read_text().splitlines()
    rec = json.loads(lines[3])
    rec["slot_id"] = "slot_3"  # someone quietly re-files an item after the fact
    lines[3] = json.dumps(rec)
    tampered.write_text("\n".join(lines) + "\n")
    print(f"tampered copy (record 3 slot_id edited): verify() -> {verify_file(tampered)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custody log tools")
    parser.add_argument("--verify", type=Path, help="verify an existing log file")
    parser.add_argument("--demo", action="store_true", help="write and verify a demo log in out/")
    args = parser.parse_args()
    if args.verify:
        bad = verify_file(args.verify)
        print(f"{args.verify}: " + ("chain intact" if bad is None else f"FIRST BAD RECORD at index {bad}"))
        raise SystemExit(0 if bad is None else 1)
    _demo(Path(__file__).resolve().parent.parent / "out")
