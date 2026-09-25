"""Fail-closed workflow tests: identity reaches the robot only through the barcode, and every refusal path refuses.

Run with `python3 tests/test_workflow.py` (or `pytest tests/`). Needs the vendored Panda model.
"""

import inspect
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import controller  # noqa: E402
import labels  # noqa: E402
import routing  # noqa: E402
import scanner  # noqa: E402
from episode import IntakeStation  # noqa: E402

_STATION = None


def station():
    global _STATION
    if _STATION is None:
        _STATION = IntakeStation(Path(tempfile.mkdtemp()) / "log.jsonl", echo=False)
    return _STATION


def _actions_since(st, n):
    return [e["action"] for e in st.log.tail(len(st.log) - n)]


def test_controller_code_has_no_path_to_true_identity():
    src = inspect.getsource(controller)
    for forbidden in ("true_item_id", "POOL_IDS", "geom_matid", "labels.", "CASE_DB", "_choose_item", "correct_location"):
        assert forbidden not in src, f"controller.py references {forbidden!r}"
    params = inspect.signature(controller.PickPlaceController.__init__).parameters
    assert not any("id" in p.lower() or "truth" in p.lower() for p in params), list(params)
    route_src = inspect.getsource(routing.route)
    assert "decoded_id" in route_src and "true_item" not in route_src and "truth" not in route_src.lower()
    assert "geom_matid" not in inspect.getsource(scanner) and "POOL_IDS" not in inspect.getsource(scanner)


def test_normal_item_is_filed_where_its_label_routes():
    st = station()
    n = len(st.log)
    r = st.run_episode(3, label=5, occupied=[])
    assert r.outcome == "filed" and r.success and not r.misfile
    assert r.decoded_id == r.true_item_id == labels.POOL_IDS[5]
    assert r.routed_location == r.correct_location == r.final_location
    assert _actions_since(st, n) == ["SUBMITTED", "REGISTERED", "PICKED", "PLACED", "VERIFIED"]
    assert r.scan["verify_match"] is True


def test_mislabelled_item_robot_follows_label_evaluator_flags_misfile():
    """World truth is a narcotics carton, but the officer's label says a weapons item. The robot can only know the
    label, so it files to CAB-B; the evaluator, judging from truth, must flag that as a misfile."""
    st = station()
    narcotics = next(i for i, r in routing.CASE_DB.items() if r.category == "narcotics")
    weapons_idx = next(k for k, i in enumerate(labels.POOL_IDS) if routing.CASE_DB[i].category == "weapons")
    real = st._choose_item
    st._choose_item = lambda seed, label: (f"label_{weapons_idx:02d}", narcotics, "carton")
    try:
        r = st.run_episode(3, occupied=[])
    finally:
        st._choose_item = real
    assert r.decoded_id == labels.POOL_IDS[weapons_idx] != r.true_item_id
    assert r.routed_location.startswith("CAB-B/") and r.correct_location.startswith("CAB-A/")
    assert r.final_location == r.routed_location and r.misfile


def _refused(kw, cause):
    st = station()
    n = len(st.log)
    r = st.run_episode(3, **kw)
    acts = _actions_since(st, n)
    assert r.outcome == "refused" and r.refusal_cause == cause and not r.misfile, (r.outcome, r.refusal_cause)
    assert r.final_location is None and "REFUSED" in acts and "VERIFIED" not in acts and "PLACED" not in acts
    return r


def test_damaged_label_is_refused_without_guessing():
    r = _refused({"label": "damaged", "occupied": []}, "no_decode")
    assert r.decoded_id is None and r.routed_location is None
    assert "intake" in r.scan and "rescan" in r.scan  # exactly one re-scan from a second pose, then refuse


def test_unregistered_label_is_refused():
    r = _refused({"label": "unregistered", "occupied": []}, "no_case_match")
    assert r.decoded_id == labels.UNREGISTERED_ID


def test_verify_mismatch_is_refused_and_item_returned():
    r = _refused({"label": 5, "occupied": [], "fault": "swap_label_in_transit"}, "verify_mismatch")
    assert r.scan["verify_match"] is False and r.grasped


def test_full_cabinet_is_refused():
    full = [f"CAB-B/slot_{i}" for i in range(4)]
    _refused({"label": 1, "occupied": full}, "cabinet_full")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
