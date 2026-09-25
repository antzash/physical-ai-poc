"""Routing fails closed: unknown IDs and full cabinets are refused, never guessed.

Run with `python3 tests/test_routing.py` (or `pytest tests/`).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import labels  # noqa: E402
import scene  # noqa: E402
from routing import CASE_DB, CATEGORY_CABINET, CabinetBank, route  # noqa: E402


def test_twelve_addresses_across_three_cabinets():
    assert len(scene.LOCATIONS) == 12
    assert {loc.split("/")[0] for loc in scene.LOCATIONS} == {"CAB-A", "CAB-B", "CAB-C"}


def test_routes_by_category_to_first_free_slot():
    bank = CabinetBank()
    for item_id, rec in list(CASE_DB.items())[:8]:
        d = route(item_id, bank)
        assert d.ok and d.cabinet == CATEGORY_CABINET[rec.category] and d.location.startswith(d.cabinet + "/")
        bank.release(d.location)


def test_unregistered_id_is_refused():
    d = route(labels.UNREGISTERED_ID, CabinetBank())
    assert not d.ok and d.refusal == "no_case_match" and d.location is None


def test_garbage_is_refused():
    for bad in ("", "EV-2026-00", "not a barcode", "EV-2026-001000 "):
        assert not route(bad, CabinetBank()).ok


def test_full_cabinet_is_refused_not_overflowed():
    bank = CabinetBank()
    weapons = [i for i, r in CASE_DB.items() if r.category == "weapons"]
    placed = [route(i, bank) for i in weapons[:4]]
    assert all(p.ok for p in placed) and len({p.location for p in placed}) == 4
    d = route(weapons[4], bank)
    assert not d.ok and d.refusal == "cabinet_full" and d.location is None


def test_reservation_blocks_double_allocation():
    bank = CabinetBank()
    ids = [i for i, r in CASE_DB.items() if r.category == "narcotics"][:2]
    a, b = route(ids[0], bank), route(ids[1], bank)
    assert a.location != b.location


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
