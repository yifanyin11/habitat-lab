import habitat_sim
import magnum as mn #graphics
import warnings
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
warnings.filterwarnings('ignore')
from habitat_sim.utils.settings import make_cfg
from matplotlib import pyplot as plt
from habitat_sim.utils import viz_utils as vut
from omegaconf import  DictConfig, OmegaConf #config manage
import numpy as np

from habitat.articulated_agents.robots import FetchRobot, StretchRobot
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import  ThirdRGBSensorConfig, HeadRGBSensorConfig, HeadPanopticSensorConfig
from habitat.config.default_structured_configs import SimulatorConfig, HabitatSimV0Config, AgentConfig

import habitat
from habitat_sim.physics import JointMotorSettings, MotionType
from habitat.config.default_structured_configs import TaskConfig, EnvironmentConfig, DatasetConfig, HabitatConfig
from habitat.config.default_structured_configs import ArmActionConfig, BaseVelocityActionConfig, OracleNavActionConfig, ActionConfig
from habitat.core.env import Env

import git, os
repo = git.Repo(".", search_parent_directories=True)
dir_path = repo.working_tree_dir
data_path = os.path.join(dir_path, "data")
os.chdir(dir_path)
output_path = os.path.join(
    dir_path, "examples/tutorials/habitat_lab_visualization/"
)
from habitat.tasks.rearrange.actions.articulated_agent_action import ArticulatedAgentAction
from habitat.core.registry import registry
from gym import spaces
def make_sim_cfg(agent_dict):
    # Start the scene config
    sim_cfg = SimulatorConfig(type="RearrangeSim-v0")
    
    # Enable Horizon Based Ambient Occlusion (HBAO) to approximate shadows.
    sim_cfg.habitat_sim_v0.enable_hbao = True
    
    sim_cfg.habitat_sim_v0.enable_physics = True

    
    # Set up an example scene
    sim_cfg.scene = os.path.join(data_path, "hab3_bench_assets/hab3-hssd/scenes/103997919_171031233.scene_instance.json")
    sim_cfg.scene_dataset = os.path.join(data_path, "hab3_bench_assets/hab3-hssd/hab3-hssd.scene_dataset_config.json")
    sim_cfg.additional_object_paths = [os.path.join(data_path, 'objects/ycb/configs/')]

    
    cfg = OmegaConf.create(sim_cfg)

    # Set the scene agents
    cfg.agents = agent_dict
    cfg.agents_order = list(cfg.agents.keys())
    return cfg

def make_hab_cfg(agent_dict, action_dict):
    sim_cfg = make_sim_cfg(agent_dict)
    task_cfg = TaskConfig(type="RearrangeEmptyTask-v0")
    task_cfg.actions = action_dict
    env_cfg = EnvironmentConfig()
    dataset_cfg = DatasetConfig(type="RearrangeDataset-v0", data_path="data/hab3_bench_assets/episode_datasets/small_large.json.gz")
    
    
    hab_cfg = HabitatConfig()
    hab_cfg.environment = env_cfg
    hab_cfg.task = task_cfg
    hab_cfg.dataset = dataset_cfg
    hab_cfg.simulator = sim_cfg
    hab_cfg.simulator.seed = hab_cfg.seed

    return hab_cfg

def init_rearrange_env(agent_dict, action_dict):
    hab_cfg = make_hab_cfg(agent_dict, action_dict)
    res_cfg = OmegaConf.create(hab_cfg)
    return Env(res_cfg)

#agent
main_agent_config = AgentConfig()
urdf_path = os.path.join(data_path, "robots/hab_stretch/urdf/hab_stretch.urdf")
main_agent_config.articulated_agent_type = "StretchRobot"
main_agent_config.articulated_agent_urdf = urdf_path

main_agent_config.sim_sensors = {
    "third_rgb": ThirdRGBSensorConfig(height=1920, width=1920),
    "head_rgb": HeadRGBSensorConfig(),
}

# import copy
# second_agent_config = copy.deepcopy(main_agent_config)
# second_agent_config.articulated_agent_type = "KinematicHumanoid"
# second_agent_config.articulated_agent_urdf = os.path.join(data_path, "hab3_bench_assets/humanoids/female_0/female_0.urdf")

# agent_dict = {"agent_0": main_agent_config, "agent_1": second_agent_config}
# action_dict = {
#     "oracle_magic_grasp_action":ArmActionConfig(type="MagicGraspAction"),
#     "base_velocity_action":BaseVelocityActionConfig(),
#     "oracle_coord_action": OracleNavActionConfig(type="OracleNavCoordinateAction", spawn_max_dist_to_obj=1.0)
# }

# multi_agent_action_dict = {}
# for action_name, action_config in action_dict.items():
#     for agent_id in range(2):
#         multi_agent_action_dict[f"agent_{agent_id}_{action_name}"] = action_config
# env = init_rearrange_env(agent_dict, multi_agent_action_dict)
import copy
second_agent_config = copy.deepcopy(main_agent_config)
second_agent_config.articulated_agent_urdf = os.path.join(data_path, "hab3_bench_assets/humanoids/female_0/female_0.urdf")
second_agent_config.articulated_agent_type = "KinematicHumanoid"
second_agent_config.motion_data_path = "data/hab3_bench_assets/humanoids/female_0/female_0_motion_data_smplx.pkl"

agent_dict = {"agent_0": main_agent_config, "agent_1": second_agent_config}
action_dict = {
    "oracle_magic_grasp_action": ArmActionConfig(type="MagicGraspAction"),
    "base_velocity_action": BaseVelocityActionConfig(),
    "oracle_coord_action": OracleNavActionConfig(type="OracleNavCoordinateAction", spawn_max_dist_to_obj=1.0)
}

multi_agent_action_dict = {}
for action_name, action_config in action_dict.items():
    for agent_id in range(2):
        multi_agent_action_dict[f"agent_{agent_id}_{action_name}"] = action_config 
env = init_rearrange_env(agent_dict, multi_agent_action_dict)

env.reset()
rom = env.sim.get_rigid_object_manager()
observations = []

# Walk towards the object

agent_displ = np.inf
agent_rot = np.inf
prev_rot = env.sim.agents_mgr[0].articulated_agent.base_rot
prev_pos = env.sim.agents_mgr[0].articulated_agent.base_pos
arm_joint = env.sim.agents_mgr[0].articulated_agent.arm_joint_pos
vals=[0.0]*len(arm_joint)
print("vals",vals)
arm_joint = env.sim.agents_mgr[0].articulated_agent.set_fixed_arm_joint_pos(vals)
print("arm_joint",arm_joint)
# for i in range(len(env.sim.agents_mgr[0].articulated_agent.arm_joint_pos)):
#     env.sim.agents_mgr[0].articulated_agent.arm_joint_pos[i] = 0.0
# arm_joint = env.sim.agents_mgr[0].articulated_agent.arm_joint_pos
# print("arm_joint",arm_joint)
while agent_displ > 1e-5 or agent_rot > 1e-5:
    prev_rot = env.sim.agents_mgr[0].articulated_agent.base_rot
    prev_pos = env.sim.agents_mgr[0].articulated_agent.base_pos
    action_dict = {
        "action": ("agent_0_oracle_coord_action", "agent_1_oracle_coord_action"), 
        "action_args": {
              "agent_0_oracle_nav_lookat_action": env.sim.agents_mgr[1].articulated_agent.base_pos,
              "agent_0_mode": 1,
              "agent_1_oracle_nav_lookat_action": env.sim.agents_mgr[1].articulated_agent.base_pos,
              "agent_1_mode": 1
          }
    }
    # env.sim.agents_mgr[1].articulated_agent.base_pos = mn.Vector3(0.005,0,0)
    observations.append(env.step(action_dict))
    
    cur_rot = env.sim.agents_mgr[0].articulated_agent.base_rot
    cur_pos = env.sim.agents_mgr[0].articulated_agent.base_pos
    agent_displ = (cur_pos - prev_pos).length()
    agent_rot = np.abs(cur_rot - prev_rot)
   
video_name = "stretch_social_nav_video"
video_path = f"{output_path}/{video_name}.mp4"

vut.make_video(
    observations,
    "agent_0_third_rgb",
    "color",
    video_path,
    open_vid=True,
    video_dims=(1080, 1080),
)