"""Headless implementation tests, not manual course-completion evidence."""
import json
from pathlib import Path
import tempfile
import unittest
import mujoco
import numpy as np
from manual_cartesian import CartesianController, Recorder, PandaPickCubeEnv


class TestManualCartesian(unittest.TestCase):
    def setUp(self):
        self.env = PandaPickCubeEnv()
        self.obs = self.env.reset(seed=42)
        self.controller = CartesianController(self.env)

    def tearDown(self):
        self.env.close()

    def test_jacobian_finite_difference(self):
        e, c = self.env, self.controller
        jp, jr = np.zeros((3, e.model.nv)), np.zeros((3, e.model.nv))
        mujoco.mj_jac(e.model, e.data, jp, jr, c.point(), c.hand)
        q = e.data.qpos.copy()
        point = c.point().copy()
        for j in range(7):
            e.data.qpos[:] = q
            e.data.qpos[j] += 1e-7
            mujoco.mj_forward(e.model, e.data)
            np.testing.assert_allclose((c.point()-point)/1e-7, jp[:, j], atol=1e-6)
        e.data.qpos[:] = q
        mujoco.mj_forward(e.model, e.data)

    def test_manual_axis_targets_and_bounded_actions(self):
        c = self.controller
        initial = c.target.copy()
        c.move(265)
        np.testing.assert_allclose(c.target-initial, [.01, 0, 0], atol=1e-12)
        c.move(264)
        c.move(263)
        np.testing.assert_allclose(c.target-initial, [0, .01, 0], atol=1e-12)
        c.move(262)
        c.move(266)
        np.testing.assert_allclose(c.target-initial, [0, 0, .01], atol=1e-12)
        for _ in range(200): c.move(266)
        self.assertEqual(c.target[2], 1.1)
        a = c.action()
        self.assertTrue(np.isfinite(a).all())
        self.assertLessEqual(abs(a[:7]).max(), .800001)
        self.assertEqual(a[7], 1)
        c.open = False
        self.assertEqual(c.action()[7], -1)

    def test_positive_x_motion_and_no_scene_position_control(self):
        e, c = self.env, self.controller
        initial = c.point().copy()
        c.move(265)
        expected = c.action().copy()
        # Change diagnostic scene coordinates only: arm controller is unaffected.
        e.data.xpos[e._cube_body_id] += [1, 2, 3]
        e.model.body_pos[e._target_body_id] += [1, 2, 3]
        np.testing.assert_array_equal(c.action(), expected)
        e.reset(seed=42)
        for _ in range(4): e.step(c.action())
        self.assertGreater(c.point()[0]-initial[0], .005)
        self.assertFalse(e._check_success())

    def test_success_refusal_and_truthful_failed_record(self):
        r = Recorder(self.env, 42, self.obs)
        r.key(265)
        self.controller.move(265)
        for _ in range(4): r.step(self.controller)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                r.save(directory, 'test_false_success', require_success=True)
            self.assertEqual(list(Path(directory).iterdir()), [])
            path = r.save(directory, 'test_failed_attempt')
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(int(data['success']), 0)
                self.assertEqual(data['obs'].shape, (4, 84, 84, 3))
                self.assertEqual(data['next_obs'].shape, (4, 84, 84, 3))
                self.assertEqual(data['actions'].shape, (4, 8))
                self.assertEqual(data['states'].shape, (5, 29))
                self.assertEqual(data['dones'].tolist(), [0, 0, 0, 1])
                self.assertFalse(data['step_success'].any())
                self.assertFalse(data['env_done'].any())
                self.assertTrue((np.diff(data['elapsed_s']) >= 0).all())
            meta = json.loads(path.with_suffix('.json').read_text())
            self.assertFalse(meta['success'])
            self.assertEqual(meta['key_events'][0]['key'], 265)
            self.assertEqual(meta['steps'], 4)
            self.assertFalse(meta['control_uses_cube_or_target_positions'])
            with self.assertRaises(ValueError): r.save(directory, 'duplicate')


if __name__ == '__main__': unittest.main(verbosity=2)
