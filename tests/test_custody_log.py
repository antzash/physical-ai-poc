"""Tamper-evidence tests for the custody log and the slot allocator.

Run with `python3 tests/test_custody_log.py` (or `pytest tests/`).
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from logger import ROBOT_ACTOR, SYSTEM_ACTOR, CustodyLog, compute_hash, officer, verify_file  # noqa: E402
from slots import CabinetFull, SlotAllocator  # noqa: E402


def _write_log(path, n_items=2):
    log = CustodyLog(path, echo=False)
    for k in range(n_items):
        item, case = f"EV-TEST-{k:04d}", "CASE-TEST-01"
        log.append(officer("1001"), "SUBMITTED", item, case, "box")
        log.append(SYSTEM_ACTOR, "REGISTERED", item, case, "box", f"slot_{k}")
        log.append(ROBOT_ACTOR, "PICKED", item, case, "box", f"slot_{k}")
        log.append(ROBOT_ACTOR, "PLACED", item, case, "box", f"slot_{k}")
    return log


def _lines(path):
    return path.read_text().splitlines()


def _rewrite(path, lines):
    path.write_text("\n".join(lines) + "\n")


def test_clean_log_verifies():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "log.jsonl"
        log = _write_log(path)
        assert len(log) == 8
        assert log.verify() is None
        first = json.loads(_lines(path)[0])
        assert first["prev_hash"] is None


def test_edited_field_is_detected_at_that_record():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "log.jsonl"
        log = _write_log(path)
        lines = _lines(path)
        rec = json.loads(lines[3])
        rec["slot_id"] = "slot_3"
        lines[3] = json.dumps(rec)
        _rewrite(path, lines)
        assert log.verify() == 3


def test_rehashed_forgery_is_detected_at_next_record():
    # A forger who edits a record and recomputes its own hash still breaks the next record's prev_hash link.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "log.jsonl"
        _write_log(path)
        lines = _lines(path)
        rec = json.loads(lines[2])
        rec["detail"] = "nothing to see here"
        rec["hash"] = compute_hash(rec)
        lines[2] = json.dumps(rec)
        _rewrite(path, lines)
        assert verify_file(path) == 3


def test_deleted_record_is_detected():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "log.jsonl"
        _write_log(path)
        lines = _lines(path)
        del lines[4]
        _rewrite(path, lines)
        assert verify_file(path) == 4


def test_reordered_records_are_detected():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "log.jsonl"
        _write_log(path)
        lines = _lines(path)
        lines[5], lines[6] = lines[6], lines[5]
        _rewrite(path, lines)
        assert verify_file(path) == 5


def test_refuses_to_append_to_broken_chain():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "log.jsonl"
        _write_log(path)
        lines = _lines(path)
        lines[1] = lines[1].replace("REGISTERED", "SUBMITTED")
        _rewrite(path, lines)
        try:
            CustodyLog(path, echo=False)
        except ValueError:
            return
        raise AssertionError("opening a tampered log for append should fail")


def test_reopen_continues_chain():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "log.jsonl"
        first = _write_log(path, n_items=1)
        second = CustodyLog(path, echo=False)
        ev = second.append(SYSTEM_ACTOR, "VERIFIED", "EV-TEST-0000", "CASE-TEST-01", "box", "slot_0")
        assert ev["prev_hash"] == first.tail(1)[0]["hash"]
        assert second.verify() is None


def test_slot_allocator_fills_then_raises():
    alloc = SlotAllocator()
    got = [alloc.allocate(f"EV-{i}") for i in range(4)]
    assert got == ["slot_0", "slot_1", "slot_2", "slot_3"]
    try:
        alloc.allocate("EV-5")
    except CabinetFull:
        pass
    else:
        raise AssertionError("fifth allocation should raise CabinetFull")
    alloc.release("slot_2")
    assert alloc.allocate("EV-6") == "slot_2"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
