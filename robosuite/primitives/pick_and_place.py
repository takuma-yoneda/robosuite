"""A primitive for pick-and-place motion.
Only supported for OSC_POSITION and OSC_POSE
"""
from copy import deepcopy
import numpy as np

from robosuite.models.grippers import gripper_model


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

class Trajectory:
    def __init__(self):
        self.clear()

    def add_transition(self, obs, rew, done, info):
        self.observations.append(obs)
        self.rewards.append(rew)
        self.dones.append(done)
        self.infos.append(info)

    def clear(self):
        self.observations = []
        self.rewards = []
        self.dones = []
        self.infos = []
        

class PickAndPlacePrimitive():
    def __init__(self, env, init_obs = None, pregrasp_height: float = 0.1, gripper_step: float = 0.05) -> None:
        self.env = env

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

        # Observation keys
        # (Pdb) p obs.keys()
        # dict_keys(['robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'frontview_image', 'cubeA_pos', 'cubeA_quat', 'cubeB_pos', 'cubeB_quat', 'gripper_to_cubeA', 'gripper_to_cubeB', 'cubeA_to_cubeB', 'robot0_proprio-state', 'object-state'])

    def env_step(self, action, obs_decorator=None):
        # TODO: save these into arrays
        obs, rew, done, info = self.env.step(action)

        if obs_decorator is not None:
            assert callable(obs_decorator)
            print('decorate obs!')
            obs = obs_decorator(obs)

        self.trajectory.add_transition(obs, rew, done, info)

        # robot0_gripper_qpos <-- what's this?
        # robot0_proprio-state <-- what's this??
        expected_obs_keys = ['robot0_eef_pos', 'robot0_eef_quat', ]
        for key in expected_obs_keys:
            assert key in obs

        # Upate gripper info
        self.gripper_site_pos = obs['robot0_eef_pos']
        self.gripper_site_quat = obs['robot0_eef_quat']

        self.prev_obs = deepcopy(obs)

        return obs, rew, done, info

    def move_pos_to(self, target_pose, pos_tolr=1e-2, rot_tolr=2e-1):
        from scipy.spatial.transform import Rotation as R
        print('move_pos_to')

        def target_reached(tgt_pose):
            tpos, tori = tgt_pose[:3], tgt_pose[3:]
            pos = self.gripper_site_pos
            pos_err = np.linalg.norm(tpos - pos)

            curr_rotmat = R.from_euler('xyz', tgt_pose[3:]).as_matrix()
            tgt_rotmat = R.from_quat(self.prev_obs['robot0_eef_quat']).as_matrix()
            # rot_err = R.from_matrix(tgt_rotmat.T @ curr_rotmat).magnitude()
            ori = R.from_matrix(tgt_rotmat.T @ curr_rotmat).as_euler('xyz')
            rot_err = abs(ori[2])

            return pos_err < pos_tolr and rot_err < rot_tolr

        done = False
        min_pos_step = 0.1
        gripper = 0.
        while not done and not target_reached(target_pose):
            to_goal = target_pose[:3] - self.gripper_site_pos
            print('error', np.linalg.norm(to_goal))
            to_goal = 5 * np.clip(np.linalg.norm(to_goal), min_pos_step, 1.) * (to_goal / np.linalg.norm(to_goal))
            tgt_rotmat = R.from_euler('xyz', target_pose[3:]).as_matrix()
            curr_rotmat = R.from_quat(self.prev_obs['robot0_eef_quat']).as_matrix()
            ori = (R.from_matrix(tgt_rotmat.T @ curr_rotmat)).as_euler('xyz')

            print('orientation', ori)
            ori[:2] = 0.
            ori = -ori
            obs, rew, done, info = self.env_step([*to_goal, *ori, gripper])

    def close_gripper(self):
        # TODO: need to know the state of gripper
        # Temporarily, just use 5 steps

        # def obs_decorator(obs):
        #     obs = deepcopy(obs)
        #     obs['frontview_image'] = add_text(obs['frontview_image'], f'grasp: {grasp_detected()}', org=(10, 10))
        #     return obs

        print('close gripper')
        def grasp_detected():
            # NOTE:
            # geom.contact_geoms --> cubeA_g0
            # gripper_geoms = self.env.robots[0].gripper.important_geoms

            # for o_geom in self.objects:
            #     print('o_geom', o_geom)

            #     # Collision with left finger pad
            #     lp_coll = self.env.check_contact(gripper_geoms['left_fingerpad'], o_geom)

            #     # Collision with right finger pad
            #     rp_coll = self.env.check_contact(gripper_geoms['right_fingerpad'], o_geom)

            #     # Collision with right finger
            #     r_coll = self.env.check_contact(gripper_geoms['right_finger'], o_geom)

            #     # Collision with left finger
            #     l_coll = self.env.check_contact(gripper_geoms['left_finger'], o_geom)

            #     print('lp_coll', lp_coll, 'rp_coll', rp_coll,
            #         'lcoll', l_coll, 'rcoll', r_coll)

            # NOTE: Explicitly list each geometry in both fingerpads.
            # left_fingerpad consists of two fingertips, and right_fingerpad has one.
            # Without this, check_grasp returns True even when an object is contact with two fingertips: one on the right_fingerpad and another on left_fingerpad.
            _gripper = self.env.robots[0].gripper
            gripper = [*_gripper.important_geoms['left_fingerpad'], *_gripper.important_geoms['right_fingerpad']]

            grasp_success = self.env._check_grasp(
                # gripper=self.env.robots[0].gripper,
                gripper=gripper,
                object_geoms=self.env.objects[0]
            )
            print('grasp success', grasp_success)
            return grasp_success
            
        done = False
        gripper_act = 1.
        num_steps = 5
        steps = 0
        while not done and not grasp_detected() and steps < num_steps:
            pos, act = np.array([0, 0, 0]), np.array([0, 0, 0])
            obs, rew, done, info = self.env_step(
                [*pos, *act, gripper_act]
            )
            # print('gripper state', obs['robot0_gripper_qpos'])
            # print('gripper state', obs['robot0_proprio-state'])
            # gripper_state = min(gripper_state + self.gripper_step, 1.0)
            # TODO: Add grasp status on obs
            steps += 1

        if grasp_detected():
            print('grasp detected!')
            return True
        return False
        # return grasp_detected()

    def open_gripper(self):
        # TODO: need to know the state of gripper
        # Temporarily, just use 5 steps

        print('open gripper')
        done = False
        gripper_act = -1
        num_steps = 5 
        steps = 0
        while not done and steps < num_steps:
            pos, act = np.array([0, 0, 0]), np.array([0, 0, 0])
            obs, rew, done, info = self.env_step([*pos, *act, gripper_act])
            # gripper_state = max(gripper_state - self.gripper_step, 0.)
            # print('gripper state', gripper_state)
            # print('gripper state', obs['robot0_gripper_qpos'])
            steps += 1

    def pick(self, target_pose):
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

        # Move to pregrasp pose
        self.move_pos_to(pregrasp_pose)
        if self.trajectory.dones[-1]:
            return deepcopy(self.trajectory), False
            
        # Move down to reach the object
        self.move_pos_to(target_pose)
        if self.trajectory.dones[-1]:
            return deepcopy(self.trajectory), False

        # Grasp
        self.close_gripper()
        if self.trajectory.dones[-1]:
            return deepcopy(self.trajectory), False

        # Move the fingers up to the pregrasp pose
        self.move_pos_to(pregrasp_pose)
        if self.trajectory.dones[-1]:
            return deepcopy(self.trajectory), False

        grasp_success = self.env._check_grasp(gripper=self.env.robots[0].gripper, object_geoms=self.env.objects)
        # Check if grasp succeeded
        # If not, return False (success = False)

        # Return trajectory history so far.
        import time
        started = time.time()
        traj = deepcopy(self.trajectory)
        elapsed = time.time() - started
        print(f'deepcopy took {elapsed:.2f} seconds')
        return traj, grasp_success

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

        # Move the gripper above the goal location
        self.move_pos_to(preplace_pose)
        if self.trajectory.dones[-1]:
            return deepcopy(self.trajectory)

        # Move down to reach the object
        self.move_pos_to(target_pose)
        if self.trajectory.dones[-1]:
            return deepcopy(self.trajectory)

        # Release grasp
        self.open_gripper()
        if self.trajectory.dones[-1]:
            return deepcopy(self.trajectory)

        # Move back to preplace pose
        self.move_pos_to(preplace_pose)
        if self.trajectory.dones[-1]:
            return deepcopy(self.trajectory)

        # Return trajectory history so far.
        import time
        started = time.time()
        traj = deepcopy(self.trajectory)
        elapsed = time.time() - started
        print(f'deepcopy took {elapsed:.2f} seconds')
        return traj


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
    controller = 'OSC_POSE'
    env = suite.make(
        env_name="Stack",
        robots="UR5e",
        gripper_types="RobotiqThreeFingerGripper",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        # camera_names=[front_cam, agent_cam],
        camera_names=['frontview', 'agentview', 'leftview', 'rightview'],
        camera_depths=True,
        camera_heights=480,
        camera_widths=640,
        controller_configs=suite.load_controller_config(default_controller=controller),
        horizon=200
    )

    obs = env.reset()
    pick_place = PickAndPlacePrimitive(env, init_obs=obs)

    pick_pos = obs['cubeA_pos']
    place_pos = obs['cubeB_pos']
    cube_halfsize = obs['cubeA_pos'][2] - pick_place.table_height
    place_pos[2] += cube_halfsize

    print('pick pos', pick_pos)
    print('place pos', place_pos)

    # pick_pose = np.array([0., 0., pick_place.table_height + 0.03, 0., 0., 0.])
    pick_pose = np.array([*pick_pos, 0., 0., 0.])
    place_pose = np.array([*place_pos, 0., 0., 0.])

    # debug
    # pick_place.move_pos_to(pick_pose)
    # observations = pick_place.trajectory.observations
    # close_traj = pick_place.close_gripper()
    # open_traj = pick_place.open_gripper()
    # observations = pick_place.trajectory.observations

    pick_trajectory, grasp_success = pick_place.pick(pick_pose)
    print('grasp success', grasp_success)
    if not pick_trajectory.dones[-1]:
        place_trajectory = pick_place.place(place_pose)
    else:
        place_trajectory = []
    observations = pick_trajectory.observations + place_trajectory.observations

    # (256, 256, 3) -> (3, 256, 256)
    # frames = [obs[f'{cam}_image'].transpose(2, 1, 0) for obs in observations]
    # frames = [np.flip(obs[f'{cam}_image'].transpose(2, 0, 1), axis=2) for obs in observations]
    cam_names = ['frontview', 'agentview', 'leftview', 'rightview']
    for cam in cam_names:
        frames = [np.flip(obs[f'{cam}_image'].transpose(2, 0, 1), axis=1) for obs in observations]
        wandb.log({cam: wandb.Video(np.asarray(frames), format='mp4', fps=20)})

    for cam in cam_names:
        frames = [(np.tile(np.flip(obs[f'{cam}_depth'].transpose(2, 0, 1), axis=1), (3, 1, 1)) * 255) for obs in observations]
        # Normalize frames
        frames = [((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8) for frame in frames]
        wandb.log({cam: wandb.Video(np.asarray(frames), format='mp4', fps=20)})
        
    wandb.finish()