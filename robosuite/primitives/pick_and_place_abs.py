"""A primitive for pick-and-place motion.
Only supported for OSC_POSITION and OSC_POSE
"""
from copy import deepcopy
import numpy as np

from robosuite.models.grippers import gripper_model
from robosuite.utils.trajectory import Trajectory

class GripperAction:
    neutral = 0.
    open = -1.
    close = .5

def add_text(img, text, **kwargs):
    import numpy as np
    import cv2
    _kwargs = {
        'org': (50, 50),
        'fontFace': 3,
        'fontScale': .6,
        'color': (0, 255, 0),
        'thickness': 1,
        **kwargs
    }
    img = cv2.putText(img=np.copy(img),
                      text=text,
                      **_kwargs)
    return img


# NOTE:
# Input: pick position & orientation (world coordiante)
# Output: place position & orientation (world coordinate)

class PickAndPlaceAbsPrimitive():
    def __init__(self, env, init_obs = None, pregrasp_height: float = 0.1, gripper_step: float = 0.05,
                 interpolate: bool = False) -> None:
        self.env = env

        # TODO: Assert that env's controller expects absolute coordinate (OSC_POSE with control_delta=False)

        # End-effector pos (?)
        self.gripper_site_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id]
        self.gripper_site_quat = env.sim.data.body_xquat[env.robots[0].eef_site_id]

        # Table height
        self.table_height = env.table_offset[2]

        # Check grasp
        # grasping_cubeA = env._check_grasp(gripper=env.robots[0].gripper, object_geoms=list_of_objects)

        self.gripper_step = gripper_step
        self.trajectory = Trajectory()
        self.pregrasp_height = pregrasp_height
        self.prev_obs = deepcopy(init_obs)

        self.interpolate = interpolate

        # Observation keys
        # (Pdb) p obs.keys()
        # dict_keys(['robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'frontview_image', 'cubeA_pos', 'cubeA_quat', 'cubeB_pos', 'cubeB_quat', 'gripper_to_cubeA', 'gripper_to_cubeB', 'cubeA_to_cubeB', 'robot0_proprio-state', 'object-state'])

    def move_to(self, target_pose, gripper_action=None, step_size=0.05):
        import math

        if gripper_action is None:
            gripper_action = self.env.robots[0].gripper.current_action

        # NOTE: currently orientation is not interpolated!!
        target_pos, target_ori = target_pose[:3], target_pose[3:]

        to_vec = target_pos - self.gripper_site_pos
        dist = np.linalg.norm(to_vec)
        num_interp_points = math.floor(dist / step_size)

        if type(gripper_action) == float:
            gripper_action = [gripper_action]

        if num_interp_points > 0:
            # Interpolate!
            normalized_to_vec = to_vec / dist
            waypoints = [(i+1) * step_size * normalized_to_vec + self.gripper_site_pos for i in range(num_interp_points)]
            waypoints += [target_pos]  # Finally reach the target pos
            for waypoint in waypoints:
                self.env_step([*waypoint, *target_ori, *gripper_action])
                if self.trajectory.dones[-1]:
                    break
            t = self.trajectory
            return (t.observations[-1], t.rewards[-1], t.dones[-1], {})
        else:
            return self.env_step([*target_pose, *gripper_action])

    def _move_gripper(self, curr_pose, tgt_gripper_action, step_size=0.4):
        import math

        if type(tgt_gripper_action) == float:
            tgt_gripper_action = [tgt_gripper_action]

        curr_gripper_action = self.env.robots[0].gripper.current_action
        diff = tgt_gripper_action - curr_gripper_action
        sign = np.sign(diff)
        num_interp_points = math.floor(diff / step_size)
        if num_interp_points > 0:
            waypoints = [(i+1) * step_size * sign + curr_gripper_action for i in range(num_interp_points)]
            waypoints += [tgt_gripper_action]
            for waypoint in waypoints:
                self.env_step([*curr_pose, *waypoint])
                if self.trajectory.dones[-1]:
                    break
            t = self.trajectory
            return (t.observations[-1], t.rewards[-1], t.dones[-1], {})
        else:
            return self.env_step([*curr_pose, *tgt_gripper_action])

    def close_gripper(self, curr_pose):
        self._move_gripper(curr_pose, GripperAction.close)

    def open_gripper(self, curr_pose):
        self._move_gripper(curr_pose, GripperAction.open)

    def get_full_trajectory(self):
        # Add the final observation to trajectory and return it
        self.trajectory.observations.append(self.prev_obs)
        return deepcopy(self.trajectory)

    def env_step(self, action, obs_decorator=None):
        # TODO: save these into arrays
        obs, rew, done, info = self.env.step(action)

        if obs_decorator is not None:
            assert callable(obs_decorator)
            obs = obs_decorator(obs)

        self.trajectory.add_transition(self.prev_obs, action, rew, done, info)

        # robot0_gripper_qpos <-- what's this?
        # robot0_proprio-state <-- what's this??
        expected_obs_keys = ['robot0_eef_pos', 'robot0_eef_quat']
        for key in expected_obs_keys:
            assert key in obs

        # Upate gripper info
        self.gripper_site_pos = obs['robot0_eef_pos']
        self.gripper_site_quat = obs['robot0_eef_quat']

        self.prev_obs = deepcopy(obs)

        return obs, rew, done, info

    def pick(self, target_pose, target_object_geoms):
        # While loop until certain height is reached
        # 1. Move the gripper above the object with specified ori
        # 2. Move down the gripper until certain height is reached
        # 3. Grip --> check if gripper succeeded
        # 4. Move the gripper up

        # NOTE: Maybe necessary: maintain the error history, and if it doesn't decrease N times in sequence, abort it
        # Make sure to show some warning.
        # It's possible that the arm is in the weird position and it cannot reach the goal.

        # TODO: reset trajectory cache
        self.trajectory.clear()

        # 1. Move the gripper to pregrasp pose
        pregrasp_pose = target_pose.copy()
        pregrasp_pose[2] = self.table_height + self.pregrasp_height

        # Set gripper to closed
        self.close_gripper(curr_pose=pregrasp_pose)

        # Move to pregrasp pose and open gripper
        if self.interpolate:
            self.move_to(pregrasp_pose)
        else:
            self.env_step([*pregrasp_pose, GripperAction.close])

        # Open gripper
        self.open_gripper(curr_pose=pregrasp_pose)
        if self.trajectory.dones[-1]:
            return self.get_full_trajectory(), False

        # Move down to reach the object
        if self.interpolate:
            self.move_to(target_pose)
        else:
            self.env_step([*target_pose, GripperAction.open])
        if self.trajectory.dones[-1]:
            return self.get_full_trajectory(), False

        # Grasp
        self.close_gripper(curr_pose=target_pose)
        if self.trajectory.dones[-1]:
            return self.get_full_trajectory(), False

        # Move the fingers up to the pregrasp pose
        if self.interpolate:
            self.move_to(pregrasp_pose)
        else:
            self.env_step([*pregrasp_pose, GripperAction.close])
        if self.trajectory.dones[-1]:
            return self.get_full_trajectory(), False

        # TODO: This can only work a single wrapper!
        try:
            _check_grasp = self.env._check_grasp
        except AttributeError:
            _check_grasp = self.env.env._check_grasp
        grasp_success = _check_grasp(gripper=self.env.robots[0].gripper, object_geoms=target_object_geoms)
            
        # Check if grasp succeeded
        # If not, return False (success = False)

        # Return trajectory history so far.
        return self.get_full_trajectory(), grasp_success

    def place(self, target_pose):
        # 1. Assert that grip is True
        # 2. Move the gripper above the goal location with specific ori
        # 3. Move down to the specific height.
        #   - If there's an object on the table, let's not push it down to the end.
        #   - -> we need to know the height of it
        self.trajectory.clear()

        # 1. Move the gripper to preplace pose
        preplace_pose = target_pose.copy()
        preplace_pose[2] = self.table_height + self.pregrasp_height

        # Gripper must be closed already
        curr_gripper_act = self.env.robots[0].gripper.current_action
        # assert abs(curr_gripper_act - GripperAction.close) < 1e-2, f'current gripper act: {curr_gripper_act} vs GripperAction.close: {GripperAction.close}'

        # Move the gripper above the goal location
        if self.interpolate:
            self.move_to(preplace_pose)
        else:
            self.env_step([*preplace_pose, GripperAction.close])
        if self.trajectory.dones[-1]:
            return self.get_full_trajectory()

        # Move down to reach the object and release gripper
        if self.interpolate:
            self.move_to(target_pose)
        else:
            self.env_step([*target_pose, GripperAction.close])

        # Open gripper
        self.open_gripper(target_pose)
        if self.trajectory.dones[-1]:
            return self.get_full_trajectory()

        # Move back to preplace pose
        if self.interpolate:
            self.move_to(preplace_pose)
        else:
            self.env_step([*preplace_pose, GripperAction.close])
        if self.trajectory.dones[-1]:
            return self.get_full_trajectory()

        # Close gripper
        self.close_gripper(preplace_pose)

        # Return trajectory history so far.
        return self.get_full_trajectory()


if __name__ == '__main__':
    import robosuite as suite
    import wandb
    wandb.login()  # NOTE: You need to set envvar WANDB_API_KEY
    wandb.init(
        # Set the project where this run will be logged
        project='robosuite-test',
        # Track hyperparameters and run metadata
        # config=vars(Args),
        # resume=True,
        # id=wandb_runid
    )

    front_cam = 'frontview'
    agent_cam = 'agentview'
    osc_pose_abs = suite.load_controller_config(default_controller='OSC_POSE_ABS')
    cam_names = ['frontview', 'agentview', 'leftview', 'rightview']
    env = suite.make(
        env_name="Stack",
        robots="UR5e",
        gripper_types="RobotiqThreeFingerAbsoluteGripper",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=cam_names,
        camera_depths=True,
        camera_heights=480,
        camera_widths=640,
        controller_configs=osc_pose_abs,
        horizon=200,
        control_freq=5
    )

    obs = env.reset()
    pick_place = PickAndPlaceAbsPrimitive(env, init_obs=obs, interpolate=True)

    pick_pos = obs['cubeA_pos']
    place_pos = obs['cubeB_pos']
    cube_halfsize = obs['cubeA_pos'][2] - pick_place.table_height
    place_pos[2] += cube_halfsize

    print('pick pos', pick_pos)
    print('place pos', place_pos)

    # TEMP
    from scipy.spatial.transform import Rotation as R
    curr_ori = R.from_quat(pick_place.gripper_site_quat).as_euler('xyz')
    # above_pose = np.array([0., 0., pick_place.table_height + 0.05, *curr_ori])
    # pick_place.env_step([*above_pose, 1.0])
    # pick_trajectory = pick_place.trajectory

    # pick_pose = np.array([0., 0., pick_place.table_height + 0.03, 0., 0., 0.])
    pick_pose = np.array([*pick_pos, *curr_ori])
    place_pose = np.array([*place_pos, *curr_ori])

    # Twist by 90 degree
    # print('curr_ori', curr_ori)
    # rx, ry, rz = curr_ori
    # print(rx, ry, rz)
    # place_pose = np.array([*place_pos, rx, ry - 0.5 * np.pi, rz])

    pick_trajectory, grasp_success = pick_place.pick(pick_pose, env.cubeA)
    

    # gripper_site_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id]
    # gripper_site_quat = env.sim.data.body_xquat[env.robots[0].eef_site_id]
    # print('gripper_site_pos', gripper_site_pos, pick_place.gripper_site_pos)
    # print('gripper_site_quat', gripper_site_quat, pick_place.gripper_site_quat)
    # rx, ry, rz = R.from_quat(gripper_site_quat).as_euler('xyz')
    # print('euler', rx, ry, rz)
    # for i in range(10):
    #     pick_place.env_step([*gripper_site_pos, rx, ry, i/10 * 2 * np.pi - np.pi, 0.])
    # observations = pick_place.trajectory.observations

    print('grasp success', grasp_success)
    if not pick_trajectory.dones[-1]:
        place_trajectory = pick_place.place(place_pose)
    else:
        place_trajectory = Trajectory()
    observations = pick_trajectory.observations + place_trajectory.observations

    # (256, 256, 3) -> (3, 256, 256)
    # frames = [obs[f'{cam}_image'].transpose(2, 1, 0) for obs in observations]
    # frames = [np.flip(obs[f'{cam}_image'].transpose(2, 0, 1), axis=2) for obs in observations]
    for cam in cam_names:
        frames = [np.flip(obs[f'{cam}_image'].transpose(2, 0, 1), axis=1) for obs in observations]
        wandb.log({cam: wandb.Video(np.asarray(frames), format='mp4', fps=4)})

    # for cam in cam_names:
    #     frames = [(np.tile(np.flip(obs[f'{cam}_depth'].transpose(2, 0, 1), axis=1), (3, 1, 1)) * 255) for obs in observations]
    #     # Normalize frames
    #     frames = [((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8) for frame in frames]
    #     wandb.log({cam: wandb.Video(np.asarray(frames), format='mp4', fps=4)})
        
    wandb.finish()