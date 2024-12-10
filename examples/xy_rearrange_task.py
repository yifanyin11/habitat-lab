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

from habitat.datasets.rearrange.samplers.receptacle import (
    OnTopOfReceptacle,
    Receptacle,
    ReceptacleSet,
    ReceptacleTracker,
    find_receptacles,
    get_navigable_receptacles,
    get_all_scenedataset_receptacles,
)


import habitat
from habitat_sim.physics import JointMotorSettings, MotionType

# from habitat_hitl.core.selection import Selection

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


@registry.register_task_action
class PickObjIdAction(ArticulatedAgentAction):
    
    @property
    def action_space(self):
        MAX_OBJ_ID = 1000
        return spaces.Dict({
            f"{self._action_arg_prefix}pick_obj_id": spaces.Discrete(MAX_OBJ_ID)
        })

    def step(self, *args, **kwargs):
        obj_id = kwargs[f"{self._action_arg_prefix}pick_obj_id"]
        print(self.cur_grasp_mgr, obj_id)
        self.cur_grasp_mgr.snap_to_obj(obj_id)

@registry.register_task_action
class DropObjIdAction(ArticulatedAgentAction):
    @property
    def action_space(self):
        MAX_OBJ_ID = 1000
        return spaces.Dict({
            f"{self._action_arg_prefix}pick_obj_id": spaces.Discrete(MAX_OBJ_ID)
        })
    def step(self, *args, **kwargs):
        obj_id = kwargs[f"{self._action_arg_prefix}drop_obj_id"]
        print(self.cur_grasp_mgr, obj_id)
        self.cur_grasp_mgr.desnap()
#env

from habitat.config.default_structured_configs import TaskConfig, EnvironmentConfig, DatasetConfig, HabitatConfig
from habitat.config.default_structured_configs import ArmActionConfig, BaseVelocityActionConfig, OracleNavActionConfig, ActionConfig
from habitat.core.env import Env
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
    dataset_cfg = DatasetConfig(type="RearrangeDataset-v0", data_path="data/hab3_bench_assets/episode_datasets/small_small.json.gz")
    
    
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

action_dict = {
    "pick_obj_id_action": ActionConfig(type="PickObjIdAction"),
    "drop_obj_id_action": ActionConfig(type="DropObjIdAction"),
    "oracle_magic_grasp_action": ArmActionConfig(type="MagicGraspAction"),
    "base_velocity_action": BaseVelocityActionConfig(),
    "oracle_coord_action": OracleNavActionConfig(type="OracleNavCoordinateAction", spawn_max_dist_to_obj=1.0)
}

#agent
main_agent_config = AgentConfig()
urdf_path = os.path.join(data_path, "robots/hab_stretch/urdf/hab_stretch.urdf")
main_agent_config.articulated_agent_type = "StretchRobot"
main_agent_config.articulated_agent_urdf = urdf_path

main_agent_config.sim_sensors = {
    "third_rgb": ThirdRGBSensorConfig(),
    "head_rgb": HeadRGBSensorConfig(),
}
agent_dict = {"main_agent": main_agent_config}

env = init_rearrange_env(agent_dict, action_dict)
# print("xytest sim", env._sim)
env.reset()

rom = env.sim.get_rigid_object_manager()
#obj_id = env.sim.scene_obj_ids[1]
obj_id = 128 #003_cracker_box_:0000 is in Vector(-4.7659, 0.56712, 1.46073)
recs = find_receptacles(env.sim)
# all_scenedataset_receptacles = get_all_scenedataset_receptacles(env.sim)
print("xytest receptable_instances")
for rec in recs:
    support_object_ids = rec.get_support_object_ids(env.sim)
    print(support_object_ids)
    # print(
    #     f" - {rec.name} : {rec.parent_object_handle} | {rec.parent_link}"
    #     )
# print("xytest get_all_scenedataset_receptacles", all_scenedataset_receptacles)
first_object = rom.get_object_by_id(obj_id)
object_trans = first_object.translation
print("obj_id",obj_id)
print(first_object.handle, "is in", object_trans)
observation = []
delta = 2.0
for i in env.sim.scene_obj_ids:
    object = rom.get_object_by_id(i)
    print("id",i, "name",object.handle)


object_agent_vec = env.sim.articulated_agent.base_pos - object_trans
object_agent_vec.y = 0
dist_agent_object = object_agent_vec.length()

agent_displ = np.inf
agent_rot = np.inf
prev_rot = env.sim.articulated_agent.base_rot
prev_pos = env.sim.articulated_agent.base_pos

while agent_displ > 1e-9 or agent_rot > 1e-9:
    prev_pos = env.sim.articulated_agent.base_pos
    prev_rot = env.sim.articulated_agent.base_rot

    action_dict = {
        "action" : ("oracle_coord_action"),
        "action_args" : {
            "oracle_nav_lookat_action": object_trans,
            "mode" : 1
        }
    }
    observation.append(env.step(action_dict))

    cur_rot = env.sim.articulated_agent.base_rot
    cur_pos = env.sim.articulated_agent.base_pos
    agent_displ = (cur_pos - prev_pos).length()
    agent_rot = np.abs(cur_rot - prev_rot)

for _ in range(50):
    action_dict = {"action":(), "action_args":{}}
    observation.append(env.step(action_dict))

action_dict = {"action":("pick_obj_id_action"), "action_args":{"pick_obj_id":obj_id}}
observation.append(env.step(action_dict))
for _ in range(50):
    action_dict = {"action": (), "action_args": {}}
    observation.append(env.step(action_dict))  

    
# Remove the object
action_dict = {"action":("drop_obj_id_action"), "action_args":{"drop_obj_id":obj_id}}
observation.append(env.step(action_dict))
for _ in range(50):
    action_dict = {"action": (), "action_args": {}}
    observation.append(env.step(action_dict))  

video_name = "stretch_rearrange_video1"
video_path = f"{output_path}/{video_name}.mp4"
vut.make_video(
    observation,
    "third_rgb",
    "color",
    video_path,
    open_vid= True,
)

