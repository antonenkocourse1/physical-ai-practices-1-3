"""Manual Cartesian keyboard recording. No expert, planner, or scene-position control."""
from pathlib import Path
import argparse
import json
import queue
import sys
import time
from datetime import datetime, timezone

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'code'))
from env import PandaPickCubeEnv

MOVES = {265: (0, 1), 264: (0, -1), 263: (1, 1), 262: (1, -1),
         266: (2, 1), 267: (2, -1),
         ord('W'): (0, 1), ord('S'): (0, -1), ord('A'): (1, 1),
         ord('D'): (1, -1), ord('E'): (2, 1), ord('Q'): (2, -1)}
HELP = '''Manual Cartesian control (world axes; NOT screen axes):
  Up/W = +X, Down/S = -X, Left/A = +Y, Right/D = -Y
  PageUp/E = +Z, PageDown/Q = -Z. Each press requests 1 cm.
  C = toggle 1 cm / 3 cm increments; Space = open/close gripper
  T = toggle top / perspective camera; B = side camera (view only)
  Period (.) = advance 10 settling steps at current manual target
  Enter = save ONLY if environment reports real success
  F = save honest failed attempt and reset; R = save aborted attempt/reset
  Escape = save unfinished attempt and quit; window close also saves
Physics is PAUSED between commands. Each move advances 4 x 0.05 s.
Success auto-saves; episode ends at the original 600 simulation-step limit.
Look at green cube/red target in viewer; no automatic pickup or target selection.
'''


class CartesianController:
    def __init__(self, env):
        self.env = env
        self.hand = env._hand_body_id
        self.orientation = env.data.xmat[self.hand].reshape(3, 3).copy()
        self.target = self.point().copy()
        self.open = True
        self.increment = 0.01

    def point(self):
        d = self.env.data
        return d.xpos[self.hand] + d.xmat[self.hand].reshape(3, 3) @ np.array([0., 0., .103])

    def move(self, key):
        axis, sign = MOVES[key]
        self.target[axis] += sign * self.increment
        # Independent fixed workspace limits; never derived from cube or target.
        self.target = np.clip(self.target, [0.10, -.60, .405], [.80, .60, 1.10])

    def action(self):
        m, d = self.env.model, self.env.data
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        point = self.point()
        mujoco.mj_jac(m, d, jp, jr, point, self.hand)
        J = np.vstack([jp[:, :7], jr[:, :7]])
        Rerr = self.orientation @ d.xmat[self.hand].reshape(3, 3).T
        rot = .5 * np.array([Rerr[2, 1]-Rerr[1, 2], Rerr[0, 2]-Rerr[2, 0], Rerr[1, 0]-Rerr[0, 1]])
        error = np.r_[2.0 * np.clip(self.target-point, -.04, .04), .6 * rot]
        dq = J.T @ np.linalg.solve(J @ J.T + .12**2 * np.eye(6), error)
        q_des = np.clip(d.qpos[:7] + np.clip(dq, -.04, .04),
                        m.actuator_ctrlrange[:7, 0], m.actuator_ctrlrange[:7, 1])
        # env.step adds .05 * action to CURRENT CTRL, not qpos.
        action = np.zeros(8, dtype=np.float32)
        action[:7] = np.clip((q_des-d.ctrl[:7])/.05, -.8, .8)
        action[7] = 1. if self.open else -1.
        return action


class Recorder:
    def __init__(self, env, seed, initial_obs):
        self.env = env
        self.seed = seed
        self.started = time.monotonic()
        self.utc = datetime.now(timezone.utc).isoformat()
        self.obs = [initial_obs.copy()]
        self.states = [env.get_privileged_state().copy()]
        self.actions, self.times, self.successes, self.terminals, self.targets = [], [], [], [], []
        self.keys = []
        self.saved = False

    def key(self, key):
        self.keys.append({'key': int(key), 'elapsed_s': time.monotonic()-self.started,
                          'before_step': len(self.actions)})

    def step(self, controller):
        action = controller.action()
        obs, success, done = self.env.step(action)
        self.actions.append(action.copy())
        self.obs.append(obs.copy())
        self.states.append(self.env.get_privileged_state().copy())
        self.targets.append(controller.target.copy())
        self.times.append(time.monotonic()-self.started)
        self.successes.append(bool(success))
        self.terminals.append(bool(done))
        return bool(success), bool(done)

    def save(self, directory, reason, require_success=False):
        real_success = bool(self.actions and self.successes[-1] and self.env._check_success())
        if require_success and not real_success:
            raise ValueError('Success save refused: environment has not confirmed success.')
        if self.saved:
            raise ValueError('This attempt was already saved.')
        if not self.actions:
            print('No simulation steps recorded; nothing saved.', flush=True)
            return None
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stem = datetime.now(timezone.utc).strftime('manual_%Y%m%dT%H%M%S_%f')
        path = directory / (stem + '.npz')
        dones = np.zeros(len(self.actions), dtype=np.float32)
        dones[-1] = 1.  # trajectory boundary; actual environment termination separately retained
        np.savez_compressed(path, obs=np.stack(self.obs[:-1]).astype(np.uint8),
            next_obs=np.stack(self.obs[1:]).astype(np.uint8),
            actions=np.stack(self.actions).astype(np.float32), dones=dones,
            success=np.int8(real_success), states=np.stack(self.states),
            manual_targets=np.stack(self.targets), elapsed_s=np.array(self.times),
            step_success=np.array(self.successes), env_done=np.array(self.terminals))
        metadata = {'source': 'manual_keyboard_cartesian', 'started_utc': self.utc,
            'seed': self.seed, 'reason': reason, 'success': real_success,
            'steps': len(self.actions), 'elapsed_s': time.monotonic()-self.started,
            'simulation_s': len(self.actions)*.05, 'key_events': self.keys,
            'outcome': self.env.get_rollout_diagnostic(),
            'control_uses_cube_or_target_positions': False,
            'physics_paused_between_commands': True}
        path.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
        self.saved = True
        print(f'SAVED {path} success={real_success} reason={reason}', flush=True)
        return path


def main():
    import mujoco.viewer
    p = argparse.ArgumentParser(description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-dir', type=Path, default=HERE/'episodes')
    args = p.parse_args()
    events = queue.Queue()
    env = PandaPickCubeEnv()
    obs = env.reset(seed=args.seed)
    controller = CartesianController(env)
    recorder = Recorder(env, args.seed, obs)
    display_data = mujoco.MjData(env.model)
    print(HELP, flush=True)
    try:
        with mujoco.viewer.launch_passive(env.model, display_data, key_callback=events.put) as viewer:
            viewer.cam.lookat[:] = [.4, 0., .48]
            viewer.cam.distance = 1.5
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
            top_view = False
            def set_camera(top=False, side=False):
                with viewer.lock():
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                    viewer.cam.lookat[:] = [.4, 0., .43] if top else [.4, 0., .48]
                    viewer.cam.distance = 1.15 if top else 1.5
                    viewer.cam.azimuth = 90 if (top or side) else 135
                    viewer.cam.elevation = -89 if top else (-15 if side else -25)
            def sync():
                display_data.qpos[:] = env.data.qpos
                display_data.qvel[:] = env.data.qvel
                mujoco.mj_forward(env.model, display_data)
                # C is also a MuJoCo visualization hotkey. Keep camera frusta hidden.
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = False
                viewer.sync()
            def reset():
                nonlocal obs, controller, recorder
                obs = env.reset(seed=args.seed)
                controller = CartesianController(env)
                recorder = Recorder(env, args.seed, obs)
                print('RESET; target at current fingertips, gripper open.', flush=True)
            sync()
            while viewer.is_running():
                try:
                    key = events.get(timeout=.05)
                except queue.Empty:
                    sync()
                    continue
                recorder.key(key)
                steps = 0
                if key in MOVES:
                    controller.move(key)
                    steps = 4
                elif key == 32:
                    controller.open = not controller.open
                    steps = 6
                elif key == ord('.'):
                    steps = 10
                elif key == ord('C'):
                    controller.increment = .03 if controller.increment == .01 else .01
                    print(f'Increment {controller.increment*100:.0f} cm', flush=True)
                elif key == ord('T'):
                    top_view = not top_view
                    set_camera(top=top_view)
                    sync()
                    print('Camera: top' if top_view else 'Camera: perspective', flush=True)
                elif key == ord('B'):
                    top_view = False
                    set_camera(side=True)
                    sync()
                    print('Camera: side', flush=True)
                elif key == 257:
                    try:
                        recorder.save(args.save_dir, 'enter_success', require_success=True)
                        reset()
                    except ValueError as e:
                        print(str(e), flush=True)
                elif key in (ord('F'), ord('R'), 256):
                    recorder.save(args.save_dir, 'failed_save' if key == ord('F') else 'aborted')
                    if key == 256:
                        break
                    reset()
                for _ in range(steps):
                    if not viewer.is_running():
                        break
                    success, done = recorder.step(controller)
                    sync()
                    time.sleep(.05)
                    if done:
                        recorder.save(args.save_dir, 'environment_success' if success else 'timeout',
                                      require_success=success)
                        reset()
                        # Discard queued repeats from the completed episode.
                        while not events.empty():
                            events.get_nowait()
                        break
                if steps:
                    print(f'step={env._step_count} target={controller.target.round(3)} '
                          f'fingertips={controller.point().round(3)} '
                          f'gripper={"OPEN" if controller.open else "CLOSED"}', flush=True)
    finally:
        if not recorder.saved and recorder.actions:
            recorder.save(args.save_dir, 'window_closed_or_interrupted')
        env.close()


if __name__ == '__main__':
    main()
