#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Manually control the articulated agent to interact with the environment. Run as
```
python examples/interative_play.py
```

By default this controls with velocity control (which makes controlling the
agent hard). To use IK control instead add the `--add-ik` command line argument.

Controls:
- For velocity control
    - 1-7 to increase the motor target for the articulated agent arm joints
    - Q-U to decrease the motor target for the articulated agent arm joints
- For IK control
    - W,S,A,D to move side to side
    - E,Q to move up and down
- I,J,K,L to move the articulated agent base around
- PERIOD to print the current world coordinates of the articulated agent base.
- Z to toggle the camera to free movement mode. When in free camera mode:
    - W,S,A,D,Q,E to translate the camera
    - I,J,K,L,U,O to rotate the camera
    - B to reset the camera position
- X to change the articulated agent that is being controlled (if there are multiple articulated agents).

Change the task with `--cfg benchmark/rearrange/close_cab.yaml` (choose any task under the `habitat-lab/habitat/config/task/rearrange/` folder).

Change the grip type:
- Suction gripper `task.actions.arm_action.grip_controller "SuctionGraspAction"`

To record a video: `--save-obs` This will save the video to file under `data/vids/` specified by `--save-obs-fname` (by default `vid.mp4`).

Record and play back trajectories:
- To record a trajectory add `--save-actions --save-actions-count 200` to
  record a truncated episode length of 200.
- By default the trajectories are saved to data/interactive_play_replays/play_actions.txt
- Play the trajectories back with `--load-actions data/interactive_play_replays/play_actions.txt`

python examples/interactive_play_cv.py --save-obs --save-obs-fname "/path/to/habitat/play.mp4"

"""

import argparse
import os
import os.path as osp
import time
from collections import defaultdict
from typing import Any, Dict, List

import magnum as mn
import numpy as np

import habitat
import habitat.tasks.rearrange.rearrange_task
from habitat.articulated_agent_controllers import HumanoidRearrangeController
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    GfxReplayMeasureMeasurementConfig,
    PddlApplyActionConfig,
    ThirdRGBSensorConfig,
    BaseVelocityActionConfig
)
from habitat.core.logging import logger
from habitat.tasks.rearrange.actions.actions import ArmEEAction
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import euler_to_quat, write_gfx_replay
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)
from habitat_sim.utils import viz_utils as vut

import cv2

# Please reach out to the paper authors to obtain this file

# CONFIG_PATH = "habitat-lab/habitat/config/benchmark/rearrange/play/play_stretch_cv.yaml"

DEFAULT_POSE_PATH = "data/humanoids/humanoid_data/walking_motion_processed.pkl"
DEFAULT_CFG = "benchmark/rearrange/play/play_stretch_cv.yaml"

DEFAULT_RENDER_STEPS_LIMIT = 60
SAVE_VIDEO_DIR = "./data/vids"
SAVE_ACTIONS_DIR = "./data/interactive_play_replays"


def step_env(env, action_name, action_args):
    return env.step({"action": action_name, "action_args": action_args})


def get_input_vel_ctlr(
    skip_pygame,
    cfg,
    arm_action,
    env,
    not_block_input,
    agent_to_control,
    control_humanoid,
    humanoid_controller,
    key=1,
):
    if skip_pygame:
        return step_env(env, "empty", {}), None, False
    multi_agent = len(env._sim.agents_mgr) > 1

    if multi_agent:
        agent_k = f"agent_{agent_to_control}_"
    else:
        agent_k = ""
    arm_action_name = f"{agent_k}arm_action"

    if control_humanoid:
        base_action_name = f"{agent_k}humanoidjoint_action"
        base_key = "human_joints_trans"
    else:
        if "spot" in cfg:
            base_action_name = f"{agent_k}base_velocity_non_cylinder"
        else:
            base_action_name = f"{agent_k}base_velocity"
        arm_key = "arm_action"
        grip_key = "grip_action"
        base_key = "base_vel"

    # if arm_action_name in env.action_space.spaces:
    #     arm_action_space = env.action_space.spaces[arm_action_name].spaces[
    #         arm_key
    #     ]
    #     arm_ctrlr = env.task.actions[arm_action_name].arm_ctrlr
    #     base_action = None
    # elif "stretch" in cfg:
    arm_action_space = np.zeros(10)
    arm_ctrlr = None
    base_action = [0, 0]
    # else:
    #     arm_action_space = np.zeros(7)
    #     arm_ctrlr = None
    #     base_action = [0, 0]

    if arm_action is None:
        arm_action = np.zeros(arm_action_space.shape[0])
        given_arm_action = False
    else:
        given_arm_action = True

    end_ep = False
    magic_grasp = None

    if key == 27:  # ESC key
        return None, None, False
    elif key == ord('m'):
        end_ep = True
    elif key == ord('n'):
        env._sim.navmesh_visualization = not env._sim.navmesh_visualization

    if not_block_input:
        # Base control
        if key == ord('j'):
            # Left
            base_action = [0, 1]
        elif key == ord('l'):
            # Right
            base_action = [0, -1]
        elif key == ord('k'):
            # Back
            base_action = [-1, 0]
        elif key == ord('i'):
            # Forward
            base_action = [1, 0]

        print(f"base_action: {base_action}")

        if arm_action_space.shape[0] == 7:
            # Velocity control. A different key for each joint
            if key == ord('q'):
                arm_action[0] = 1.0
            elif key == ord('1'):
                arm_action[0] = -1.0

            elif key == ord('w'):
                arm_action[1] = 1.0
            elif key == ord('2'):
                arm_action[1] = -1.0

            elif key == ord('e'):
                arm_action[2] = 1.0
            elif key == ord('3'):
                arm_action[2] = -1.0

            elif key == ord('r'):
                arm_action[3] = 1.0
            elif key == ord('4'):
                arm_action[3] = -1.0

            elif key == ord('t'):
                arm_action[4] = 1.0
            elif key == ord('5'):
                arm_action[4] = -1.0

            elif key == ord('y'):
                arm_action[5] = 1.0
            elif key == ord('6'):
                arm_action[5] = -1.0

            elif key == ord('u'):
                arm_action[6] = 1.0
            elif key == ord('7'):
                arm_action[6] = -1.0

            elif key == ord('t'):
                arm_action[7] = 1.0
            elif key == ord('5'):
                arm_action[7] = -1.0

            elif key == ord('y'):
                arm_action[8] = 1.0
            elif key == ord('6'):
                arm_action[8] = -1.0

            elif key == ord('u'):
                arm_action[9] = 1.0
            elif key == ord('7'):
                arm_action[9] = -1.0

        elif arm_action_space.shape[0] == 10:
            # Velocity control. A different key for each joint
            if key == ord('q'):
                arm_action[0] = 1.0
            elif key == ord('1'):
                arm_action[0] = -1.0

            elif key == ord('w'):
                arm_action[4] = 1.0
            elif key == ord('2'):
                arm_action[4] = -1.0

            elif key == ord('e'):
                arm_action[5] = 1.0
            elif key == ord('3'):
                arm_action[5] = -1.0

            elif key == ord('r'):
                arm_action[6] = 1.0
            elif key == ord('4'):
                arm_action[6] = -1.0

            elif key == ord('t'):
                arm_action[7] = 1.0
            elif key == ord('5'):
                arm_action[7] = -1.0

            elif key == ord('y'):
                arm_action[8] = 1.0
            elif key == ord('6'):
                arm_action[8] = -1.0

            elif key == ord('u'):
                arm_action[9] = 1.0
            elif key == ord('7'):
                arm_action[9] = -1.0

        elif isinstance(arm_ctrlr, ArmEEAction):
            EE_FACTOR = 0.5
            # End effector control
            if key == ord('d'):
                arm_action[1] -= EE_FACTOR
            elif key == ord('a'):
                arm_action[1] += EE_FACTOR
            elif key == ord('w'):
                arm_action[0] += EE_FACTOR
            elif key == ord('s'):
                arm_action[0] -= EE_FACTOR
            elif key == ord('q'):
                arm_action[2] += EE_FACTOR
            elif key == ord('e'):
                arm_action[2] -= EE_FACTOR
        else:
            raise ValueError("Unrecognized arm action space")

        if key == ord('p'):
            logger.info("[play.py]: Unsnapping")
            # Unsnap
            magic_grasp = -1
        elif key == ord('o'):
            # Snap
            logger.info("[play.py]: Snapping")
            magic_grasp = 1

    if control_humanoid:
        if humanoid_controller is None:
            # Add random noise to human arms but keep global transform
            (
                joint_trans,
                root_trans,
            ) = env._sim.articulated_agent.get_joint_transform()
            # Divide joint_trans by 4 since joint_trans has flattened quaternions
            # and the dimension of each quaternion is 4
            num_joints = len(joint_trans) // 4
            root_trans = np.array(root_trans)
            index_arms_start = 10
            joint_trans_quat = [
                mn.Quaternion(
                    mn.Vector3(joint_trans[(4 * index) : (4 * index + 3)]),
                    joint_trans[4 * index + 3],
                )
                for index in range(num_joints)
            ]
            rotated_joints_quat = []
            for index, joint_quat in enumerate(joint_trans_quat):
                random_vec = np.random.rand(3)
                # We allow for maximum 10 angles per step
                random_angle = np.random.rand() * 10
                rotation_quat = mn.Quaternion.rotation(
                    mn.Rad(random_angle), mn.Vector3(random_vec).normalized()
                )
                if index > index_arms_start:
                    joint_quat *= rotation_quat
                rotated_joints_quat.append(joint_quat)
            joint_trans = np.concatenate(
                [
                    np.array(list(quat.vector) + [quat.scalar])
                    for quat in rotated_joints_quat
                ]
            )
            base_action = np.concatenate(
                [joint_trans.reshape(-1), root_trans.transpose().reshape(-1)]
            )
        else:
            # Use the controller
            relative_pos = mn.Vector3(base_action[0], 0, base_action[1])
            humanoid_controller.calculate_walk_pose(relative_pos)
            base_action = humanoid_controller.get_pose()

    if key == ord('.'):
        # Print the current position of the articulated agent, useful for debugging.
        pos = [
            float("%.3f" % x)
            for x in env._sim.articulated_agent.sim_obj.translation
        ]
        rot = env._sim.articulated_agent.sim_obj.rotation
        ee_pos = env._sim.articulated_agent.ee_transform().translation
        logger.info(
            f"Agent state: pos = {pos}, rotation = {rot}, ee_pos = {ee_pos}"
        )
    if key == ord(','):
        # Print the current arm state of the articulated agent, useful for debugging.
        joint_state = [
            float("%.3f" % x) for x in env._sim.articulated_agent.arm_joint_pos
        ]
        logger.info(f"Agent arm joint state: {joint_state}")

    args: Dict[str, Any] = {}

    print('base_action is not None', base_action is not None)
    print('base_action_name in env.action_space.spaces', base_action_name in env.action_space.spaces)

    # if base_action is not None and base_action_name in env.action_space.spaces:
    if base_action != [0, 0]:
        name = base_action_name
        args = {base_key: base_action}
    else:
        name = arm_action_name
        if given_arm_action:
            # The grip is also contained in the provided action
            args = {
                arm_key: arm_action[:-1],
                grip_key: arm_action[-1],
            }
        else:
            args = {arm_key: arm_action, grip_key: magic_grasp}

    if magic_grasp is None:
        arm_action = [*arm_action, 0.0]
    else:
        arm_action = [*arm_action, magic_grasp]

    return step_env(env, name, args), arm_action, end_ep


def get_wrapped_prop(venv, prop):
    if hasattr(venv, prop):
        return getattr(venv, prop)
    elif hasattr(venv, "venv"):
        return get_wrapped_prop(venv.venv, prop)
    elif hasattr(venv, "env"):
        return get_wrapped_prop(venv.env, prop)

    return None


class FreeCamHelper:
    def __init__(self):
        self._is_free_cam_mode = False
        self._last_pressed = 0
        self._free_rpy = np.zeros(3)
        self._free_xyz = np.zeros(3)

    @property
    def is_free_cam_mode(self):
        return self._is_free_cam_mode

    def update(self, env, step_result, update_idx, key=None):
        if key == ord('z') and (update_idx - self._last_pressed) > 60:
            self._is_free_cam_mode = not self._is_free_cam_mode
            logger.info(f"Switching camera mode to {self._is_free_cam_mode}")
            self._last_pressed = update_idx

        if self._is_free_cam_mode:
            offset_rpy = np.zeros(3)
            if key == ord('u'):
                offset_rpy[1] += 1
            elif key == ord('o'):
                offset_rpy[1] -= 1
            elif key == ord('i'):
                offset_rpy[2] += 1
            elif key == ord('k'):
                offset_rpy[2] -= 1
            elif key == ord('j'):
                offset_rpy[0] += 1
            elif key == ord('l'):
                offset_rpy[0] -= 1

            offset_xyz = np.zeros(3)
            if key == ord('q'):
                offset_xyz[1] += 1
            elif key == ord('e'):
                offset_xyz[1] -= 1
            elif key == ord('w'):
                offset_xyz[2] += 1
            elif key == ord('s'):
                offset_xyz[2] -= 1
            elif key == ord('a'):
                offset_xyz[0] += 1
            elif key == ord('d'):
                offset_xyz[0] -= 1
            offset_rpy *= 0.1
            offset_xyz *= 0.1
            self._free_rpy += offset_rpy
            self._free_xyz += offset_xyz
            if key == ord('b'):
                self._free_rpy = np.zeros(3)
                self._free_xyz = np.zeros(3)

            quat = euler_to_quat(self._free_rpy)
            trans = mn.Matrix4.from_(
                quat.to_matrix(), mn.Vector3(*self._free_xyz)
            )
            env._sim._sensors[
                "third_rgb"
            ]._sensor_object.node.transformation = trans
            step_result = env._sim.get_sensor_observations()
            return step_result
        return step_result


def play_env(env, args, config):
    render_steps_limit = None
    if args.no_render:
        render_steps_limit = DEFAULT_RENDER_STEPS_LIMIT

    use_arm_actions = None
    if args.load_actions is not None:
        with open(args.load_actions, "rb") as f:
            use_arm_actions = np.load(f)
            logger.info("Loaded arm actions")

    obs = env.reset()

    if not args.no_render:
        draw_obs = observations_to_image(obs, {})
        draw_obs = cv2.cvtColor(draw_obs, cv2.COLOR_RGB2BGR)
        # Initialize OpenCV window
        cv2.namedWindow('Habitat', cv2.WINDOW_NORMAL)
        cv2.imshow('Habitat', draw_obs)

    update_idx = 0
    target_fps = 60.0
    prev_time = time.time()
    all_obs = []
    total_reward = 0
    all_arm_actions: List[float] = []
    agent_to_control = 0

    free_cam = FreeCamHelper()
    gfx_measure = env.task.measurements.measures.get(
        GfxReplayMeasure.cls_uuid, None
    )
    is_multi_agent = len(env._sim.agents_mgr) > 1

    humanoid_controller = None
    if args.use_humanoid_controller:
        humanoid_controller = HumanoidRearrangeController(args.walk_pose_path)
        humanoid_controller.reset(env._sim.articulated_agent.base_pos)

    while True:
        if (
            args.save_actions
            and len(all_arm_actions) > args.save_actions_count
        ):
            # quit the application when the action recording queue is full
            break
        if render_steps_limit is not None and update_idx > render_steps_limit:
            break

        if not args.no_render:
            # Update OpenCV window
            cv2.imshow('Habitat', draw_obs)
            key = cv2.waitKey(1) & 0xFF

        if not args.no_render and is_multi_agent and key == ord('x'):
            agent_to_control += 1
            agent_to_control = agent_to_control % len(env._sim.agents_mgr)
            logger.info(
                f"Controlled agent changed. Controlling agent {agent_to_control}."
            )

        step_result, arm_action, end_ep = get_input_vel_ctlr(
            args.no_render,
            args.cfg,
            use_arm_actions[update_idx]
            if use_arm_actions is not None
            else None,
            env,
            not free_cam.is_free_cam_mode,
            agent_to_control,
            args.control_humanoid,
            humanoid_controller=humanoid_controller,
            key=key
        )

        if not args.no_render and key == ord('c'):
            pddl_action = env.task.actions["pddl_apply_action"]
            logger.info("Actions:")
            actions = pddl_action._action_ordering
            for i, action in enumerate(actions):
                logger.info(f"{i}: {action}")
            entities = pddl_action._entities_list
            logger.info("Entities")
            for i, entity in enumerate(entities):
                logger.info(f"{i}: {entity}")
            action_sel = input("Enter Action Selection: ")
            entity_sel = input("Enter Entity Selection: ")
            action_sel = int(action_sel)
            entity_sel = [int(x) + 1 for x in entity_sel.split(",")]
            ac = np.zeros(pddl_action.action_space["pddl_action"].shape[0])
            ac_start = pddl_action.get_pddl_action_start(action_sel)
            ac[ac_start : ac_start + len(entity_sel)] = entity_sel

            step_env(env, "pddl_apply_action", {"pddl_action": ac})

        if not args.no_render and key == ord('g'):
            pred_list = env.task.sensor_suite.sensors[
                "all_predicates"
            ]._predicates_list
            pred_values = step_result["all_predicates"]
            logger.info("\nPredicate Truth Values:")
            for i, (pred, pred_value) in enumerate(
                zip(pred_list, pred_values)
            ):
                logger.info(f"{i}: {pred.compact_str} = {pred_value}")

        if step_result is None:
            break

        if end_ep:
            total_reward = 0
            # Clear the saved keyframes.
            if gfx_measure is not None:
                gfx_measure.get_metric(force_get=True)
            env.reset()

        if not args.no_render:
            step_result = free_cam.update(env, step_result, update_idx, key=key)

        all_arm_actions.append(arm_action)
        update_idx += 1
        if use_arm_actions is not None and update_idx >= len(use_arm_actions):
            break

        obs = step_result
        info = env.get_metrics()
        reward_key = [k for k in info if "reward" in k]
        if len(reward_key) > 0:
            reward = info[reward_key[0]]
        else:
            reward = 0.0

        total_reward += reward
        info["Total Reward"] = total_reward

        if free_cam.is_free_cam_mode:
            cam = obs["third_rgb"]
            use_ob = np.zeros(draw_obs.shape)
            use_ob[:, : cam.shape[1]] = cam[:, :, :3]

        else:
            # print(obs.keys())
            use_ob = observations_to_image(obs, info)
            use_ob = cv2.cvtColor(use_ob, cv2.COLOR_RGB2BGR)
            if not args.skip_render_text:
                use_ob = overlay_frame(use_ob, info)

        draw_obs = use_ob[:]

        # if not args.no_render:
        #     draw_obs = np.transpose(draw_obs, (1, 0, 2))

        if args.save_obs:
            draw_obs_save = np.transpose(draw_obs, (1, 0, 2))
            draw_obs_save = cv2.cvtColor(draw_obs_save, cv2.COLOR_RGB2BGR)
            all_obs.append(draw_obs_save)  # type: ignore[assignment]

        if env.episode_over:
            total_reward = 0
            env.reset()

        curr_time = time.time()
        diff = curr_time - prev_time
        delay = max(1.0 / target_fps - diff, 0)
        time.sleep(delay)
        prev_time = curr_time

    if args.save_actions:
        if len(all_arm_actions) < args.save_actions_count:
            raise ValueError(
                f"Only did {len(all_arm_actions)} actions but {args.save_actions_count} are required"
            )
        all_arm_actions = all_arm_actions[: args.save_actions_count]
        os.makedirs(SAVE_ACTIONS_DIR, exist_ok=True)
        save_path = osp.join(SAVE_ACTIONS_DIR, args.save_actions_fname)
        with open(save_path, "wb") as f:
            np.save(f, all_arm_actions)
        logger.info(f"Saved actions to {save_path}")
        cv2.destroyAllWindows()
        return

    if args.save_obs:
        all_obs = np.array(all_obs)  # type: ignore[assignment]
        all_obs = np.transpose(all_obs, (0, 2, 1, 3))  # type: ignore[assignment]
        os.makedirs(SAVE_VIDEO_DIR, exist_ok=True)
        vut.make_video(
            np.expand_dims(all_obs, 1),
            0,
            "color",
            osp.join(SAVE_VIDEO_DIR, args.save_obs_fname),
        )
    if gfx_measure is not None:
        gfx_str = gfx_measure.get_metric(force_get=True)
        write_gfx_replay(
            gfx_str, config.habitat.task, env.current_episode.episode_id
        )

    if not args.no_render:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render", action="store_true", default=False)
    parser.add_argument("--save-obs", action="store_true", default=False)
    parser.add_argument("--save-obs-fname", type=str, default="play.mp4")
    parser.add_argument("--save-actions", action="store_true", default=False)
    parser.add_argument(
        "--save-actions-fname", type=str, default="play_actions.txt"
    )
    parser.add_argument(
        "--save-actions-count",
        type=int,
        default=200,
        help="""
            The number of steps the saved action trajectory is clipped to. NOTE
            the episode must be at least this long or it will terminate with
            error.
            """,
    )
    parser.add_argument("--play-cam-res", type=int, default=512)
    parser.add_argument(
        "--skip-render-text", action="store_true", default=False
    )
    parser.add_argument(
        "--same-task",
        action="store_true",
        default=False,
        help="If true, then do not add the render camera for better visualization",
    )
    parser.add_argument(
        "--skip-task",
        action="store_true",
        default=False,
        help="If true, then do not add the render camera for better visualization",
    )
    parser.add_argument(
        "--never-end",
        action="store_true",
        default=False,
        help="If true, make the task never end due to reaching max number of steps",
    )
    parser.add_argument(
        "--disable-inverse-kinematics",
        action="store_true",
        help="If specified, does not add the inverse kinematics end-effector control.",
    )

    parser.add_argument(
        "--control-humanoid",
        action="store_true",
        default=False,
        help="Control humanoid agent.",
    )

    parser.add_argument(
        "--use-humanoid-controller",
        action="store_true",
        default=False,
        help="Control humanoid agent.",
    )

    parser.add_argument(
        "--gfx",
        action="store_true",
        default=False,
        help="Save a GFX replay file.",
    )
    parser.add_argument("--load-actions", type=str, default=None)
    parser.add_argument("--cfg", type=str, default=DEFAULT_CFG)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--walk-pose-path", type=str, default=DEFAULT_POSE_PATH
    )

    args = parser.parse_args()

    config = habitat.get_config(args.cfg, args.opts)
    with habitat.config.read_write(config):
        env_config = config.habitat.environment
        sim_config = config.habitat.simulator
        task_config = config.habitat.task

        if not args.same_task:
            sim_config.debug_render = True
            agent_config = get_agent_config(sim_config=sim_config)
            agent_config.sim_sensors.update(
                {
                    "third_rgb_sensor": ThirdRGBSensorConfig(
                        height=args.play_cam_res, width=args.play_cam_res
                    )
                }
            )
            if "pddl_success" in task_config.measurements:
                task_config.measurements.pddl_success.must_call_stop = False
            if "rearrange_nav_to_obj_success" in task_config.measurements:
                task_config.measurements.rearrange_nav_to_obj_success.must_call_stop = (
                    False
                )
            if "force_terminate" in task_config.measurements:
                task_config.measurements.force_terminate.max_accum_force = -1.0
                task_config.measurements.force_terminate.max_instant_force = (
                    -1.0
                )

        if args.gfx:
            sim_config.habitat_sim_v0.enable_gfx_replay_save = True
            task_config.measurements.update(
                {"gfx_replay_measure": GfxReplayMeasureMeasurementConfig()}
            )

        if args.never_end:
            env_config.max_episode_steps = 0

        if args.control_humanoid:
            args.disable_inverse_kinematics = True

        if not args.disable_inverse_kinematics:
            if "arm_action" not in task_config.actions:
                raise ValueError(
                    "Action space does not have any arm control so cannot add inverse kinematics. Specify the `--disable-inverse-kinematics` option"
                )
            sim_config.agents.main_agent.ik_arm_urdf = (
                # "./data/robots/hab_fetch/robots/fetch_onlyarm.urdf"
                "./data/robots/hab_stretch/urdf/hab_stretch.urdf"
            )
            task_config.actions.arm_action.arm_controller = "ArmRelPosKinematicReducedActionStretch"
            task_config.actions.base_velocity = BaseVelocityActionConfig
            task_config.actions.arm_action.grip_controller = "MagicGraspAction"
        if task_config.type == "RearrangePddlTask-v0":
            task_config.actions["pddl_apply_action"] = PddlApplyActionConfig()

    with habitat.Env(config=config) as env:
        play_env(env, args, config)
