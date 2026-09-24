"""Damped least-squares inverse kinematics for the Panda `tcp` site."""

from dataclasses import dataclass

import mujoco
import numpy as np

import scene


@dataclass
class IKResult:
    q: np.ndarray
    pos_err: float
    rot_err: float
    converged: bool
    iters: int


class IK:
    """Holds a scratch MjData so solving never disturbs the simulation state."""

    def __init__(self, model, site="tcp", damping=0.15, max_step=0.1, pos_tol=5e-4, rot_tol=5e-3, max_iters=150):
        self.model = model
        self.scratch = mujoco.MjData(model)
        self.site = scene.site_id(model, site)
        self.qadr = scene.arm_qpos_adr(model)
        self.dadr = scene.arm_dof_adr(model)
        self.ranges = scene.arm_ranges(model)
        self.damping = damping
        self.max_step = max_step
        self.pos_tol = pos_tol
        self.rot_tol = rot_tol
        self.max_iters = max_iters
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def solve(self, data, target_pos, target_quat, q_init=None) -> IKResult:
        m, s = self.model, self.scratch
        s.qpos[:] = data.qpos
        if q_init is not None:
            s.qpos[self.qadr] = q_init
        target_pos = np.asarray(target_pos, dtype=float)
        target_quat = np.asarray(target_quat, dtype=float)
        cur_quat, neg_quat, dq_quat = np.zeros(4), np.zeros(4), np.zeros(4)
        rot_vec = np.zeros(3)
        lam2 = self.damping**2
        pos_err = rot_err = np.inf
        for it in range(1, self.max_iters + 1):
            mujoco.mj_kinematics(m, s)
            mujoco.mj_comPos(m, s)
            err_pos = target_pos - s.site_xpos[self.site]
            mujoco.mju_mat2Quat(cur_quat, s.site_xmat[self.site])
            mujoco.mju_negQuat(neg_quat, cur_quat)
            mujoco.mju_mulQuat(dq_quat, target_quat, neg_quat)
            mujoco.mju_quat2Vel(rot_vec, dq_quat, 1.0)
            pos_err, rot_err = np.linalg.norm(err_pos), np.linalg.norm(rot_vec)
            if pos_err < self.pos_tol and rot_err < self.rot_tol:
                return IKResult(s.qpos[self.qadr].copy(), pos_err, rot_err, True, it)
            mujoco.mj_jacSite(m, s, self._jacp, self._jacr, self.site)
            J = np.vstack([self._jacp[:, self.dadr], self._jacr[:, self.dadr]])
            err = np.concatenate([err_pos, rot_vec])
            dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), err)
            norm = np.linalg.norm(dq)
            if norm > self.max_step:
                dq *= self.max_step / norm
            q = s.qpos[self.qadr] + dq
            s.qpos[self.qadr] = np.clip(q, self.ranges[:, 0], self.ranges[:, 1])
        return IKResult(s.qpos[self.qadr].copy(), pos_err, rot_err, False, self.max_iters)


def solve_ik(model, data, target_pos, target_quat):
    """Convenience wrapper returning only the arm joint vector (see IK.solve for convergence info)."""
    return IK(model).solve(data, target_pos, target_quat).q


def down_quat(yaw):
    """TCP orientation pointing straight down, with the finger closing axis at world angle `yaw`."""
    c, s = np.cos(yaw), np.sin(yaw)
    # Columns are the TCP frame axes in world coordinates: x = y_axis x z_axis, y = closing axis, z = down.
    mat = np.array([[-s, c, 0.0], [c, s, 0.0], [0.0, 0.0, -1.0]])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.flatten())
    return quat


def tcp_yaw(data, site):
    """World angle of the TCP's finger closing axis (its y axis) projected on the horizontal plane."""
    mat = data.site_xmat[site].reshape(3, 3)
    return float(np.arctan2(mat[1, 1], mat[0, 1]))


def nearest_equivalent_yaw(yaw, reference):
    """The gripper is symmetric under a half turn, so pick yaw + k*pi closest to `reference`."""
    return reference + ((yaw - reference + np.pi / 2) % np.pi) - np.pi / 2
