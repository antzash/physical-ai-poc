"""Tests for the perception seam: noise reaches the controller, and only the controller.

Run with `python3 tests/test_perception.py` (or `pytest tests/`). Needs the vendored Panda model.
"""

import inspect
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import controller  # noqa: E402
import episode  # noqa: E402
import evaluate  # noqa: E402
import perception  # noqa: E402
import scene  # noqa: E402
from perception import NoiseSpec, PerceptionError, draw_error, observe  # noqa: E402
from randomise import Randomiser  # noqa: E402


def _placed_scene(seed=1000):
    m, d = scene.load()
    scene.reset_home(m, d)
    params = Randomiser(m).apply(d, seed)
    cls = params.object_class
    return m, d, cls, scene.body_id(m, scene.item_body(cls)), scene.geom_id(m, scene.item_geom(cls))


def _kind(cls):
    return "cylinder" if cls == "cylinder" else "other"


def test_zero_error_observation_is_ground_truth():
    m, d, cls, b, g = _placed_scene()
    obs = observe(m, d, b, g, perception.ZERO_ERROR, _kind(cls))
    R = d.xmat[b].reshape(3, 3)
    assert np.array_equal(obs.position, d.xpos[b])
    assert obs.yaw == float(np.arctan2(R[1, 0], R[0, 0]))


def test_error_is_applied_identically_on_every_read():
    m, d, cls, b, g = _placed_scene()
    err = draw_error(NoiseSpec(0.01, np.radians(5), 0.05), np.random.default_rng(3))
    first = observe(m, d, b, g, err, _kind(cls))
    second = observe(m, d, b, g, err, _kind(cls))
    assert np.array_equal(first.position, second.position) and first.yaw == second.yaw
    assert np.allclose(first.position - d.xpos[b], err.pos_offset)


def test_noise_never_moves_the_item():
    m, d, cls, b, g = _placed_scene()
    qpos = d.qpos.copy()
    err = draw_error(NoiseSpec(0.05, np.radians(30), 0.2), np.random.default_rng(0))
    observe(m, d, b, g, err, _kind(cls))
    assert np.array_equal(d.qpos, qpos)


def test_error_is_drawn_exactly_once_per_episode():
    calls = []
    real = episode.draw_error

    def counting(spec, rng):
        calls.append(1)
        return real(spec, rng)

    episode.draw_error = counting
    try:
        with tempfile.TemporaryDirectory() as tmp:
            st = episode.IntakeStation(Path(tmp) / "log.jsonl", echo=False)
            st.run_episode(1000, slot="slot_0", noise=NoiseSpec(0.005, np.radians(5)))
    finally:
        episode.draw_error = real
    assert len(calls) == 1


def test_noise_reaches_the_controller_target():
    """With a known +20 mm x offset the controller's APPROACH goal is displaced by exactly that offset."""
    m, d, cls, b, g = _placed_scene(seed=1007)  # a box
    true_xy = d.xpos[b][:2].copy()
    err = PerceptionError(pos_offset=(0.02, 0.0, 0.0))
    c = controller.PickPlaceController(m, d, cls, "slot_0", perception_error=err)
    while c.phase != "APPROACH":
        c.update()
        mujoco.mj_step(m, d)
    assert np.allclose(c.goal[0][:2], true_xy + [0.02, 0.0], atol=1e-9)


def test_evaluator_judges_on_ground_truth():
    """A 60 mm belief error makes the robot miss. The evaluator must report failure from the true state, and the
    placement error of a successful run must equal the true distance to the slot centre."""
    with tempfile.TemporaryDirectory() as tmp:
        st = episode.IntakeStation(Path(tmp) / "log.jsonl", echo=False)
        baseline = st.run_episode(1007, slot="slot_0", noise=NoiseSpec())
        assert baseline.success  # sanity: this seed succeeds with perfect state
        episode_draw = episode.draw_error
        episode.draw_error = lambda spec, rng: PerceptionError(pos_offset=(0.06, 0.0, 0.0))
        try:
            miss = st.run_episode(1007, slot="slot_0", noise=NoiseSpec(0.06))
        finally:
            episode.draw_error = episode_draw
        assert not miss.success and miss.failure_phase in ("CLOSE", "LIFT", "DESCEND")
        actions = [e["action"] for e in st.log.tail(3)]
        assert "PLACED" not in actions and "VERIFIED" not in actions and actions[-1] == "FAILED"

        ok = st.run_episode(1007, slot="slot_1", noise=NoiseSpec(0.002))
        center, _ = scene.slot_volume(st.m, st.d, "slot_1")
        true_err = float(np.linalg.norm(st.d.xpos[scene.body_id(st.m, "item_box")][:2] - center[:2]))
        assert ok.success and abs(ok.placement_error - round(true_err, 4)) < 1e-9


def test_evaluator_code_never_touches_the_observation():
    for fn in (controller.grasp_ok, controller.grasp_sample, controller.grasp_verified,
               controller.finger_contact_forces, controller.PickPlaceController._check_hold):
        src = inspect.getsource(fn)
        assert "observe" not in src and "perception" not in src and "obs" not in src, fn.__name__
    assert "observe(" not in inspect.getsource(evaluate)
    # The judging block (VERIFY, placement error) runs between the end of the cycle and the result record.
    src = inspect.getsource(episode.IntakeStation.run_episode)
    judging = src.split("r = c.result", 1)[1].split("return EpisodeResult", 1)[0]
    assert "observe(" not in judging and "perception_error" not in judging and "d.xpos[item_body]" in judging


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
