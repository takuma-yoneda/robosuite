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
    def __init__(self, env, gripper_step: int = 0.05) -> None:
        self.env = env

        # End-effector pos (?)
        self.gripper_site_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id]

        # Table height
        self.table_height = env.table_offset[2]

        # Check grasp
        self.objects = [env.cubeA, env.cubeB]
        # grasping_cubeA = env._check_grasp(gripper=env.robots[0].gripper, object_geoms=list_of_objects)

        self.gripper_step = gripper_step
        self.trajectory = Trajectory()
        self.gripper_state = 0.

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

        return obs, rew, done, info

    def move_pos_to(self, target_pose, error=1e-2):
        print('move_pos_to')

        def target_reached(tgt_pose):
            tpos, tori = tgt_pose[:3], tgt_pose[3:]
            pos = self.gripper_site_pos
            return np.linalg.norm(tpos - pos) < error

        done = False
        min_pos_step = 0.1
        while not done and not target_reached(target_pose):
            to_goal = target_pose[:3] - self.gripper_site_pos
            print('error', np.linalg.norm(to_goal))
            to_goal = 5 * np.clip(np.linalg.norm(to_goal), min_pos_step, 1.) * (to_goal / np.linalg.norm(to_goal))
            ori = np.zeros(3)
            gripper = self.gripper_state
            obs, rew, done, info = self.env_step([*to_goal, *ori, gripper])

    def close_gripper(self):
        # TODO: need to know the state of gripper
        # Temporarily, just use 5 steps

        def obs_decorator(obs):
            obs = deepcopy(obs)
            obs['frontview_image'] = add_text(obs['frontview_image'], f'grasp: {grasp_detected()}', org=(10, 10))
            return obs

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

            grasp_success = self.env._check_grasp(
                gripper=self.env.robots[0].gripper,
                object_geoms=self.objects[0]
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
                [*pos, *act, gripper_act],
                obs_decorator=obs_decorator
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
        pregrasp_pose[2] = self.table_height + 0.2


        # Move to pregrasp pose
        self.move_pos_to(pregrasp_pose)
        if self.trajectory.dones[-1]:
            return deepcopy(self.trajectory)
            

        # Move down to reach the object
        self.move_pos_to(target_pose)
        if self.trajectory.dones[-1]:
            return deepcopy(self.trajectory)

        # Grasp
        self.close_gripper()
        if self.trajectory.dones[-1]:
            return deepcopy(self.trajectory)

        # Move the fingers up to the pregrasp pose
        self.move_pos_to(pregrasp_pose)
        if self.trajectory.dones[-1]:
            return deepcopy(self.trajectory)

        grasp_success = self.env._check_grasp(gripper=self.env.robots[0].gripper, object_geoms=self.objects)
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
        preplace_pose[2] = self.table_height + 0.2

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
        camera_names='all-robotview',
        controller_configs=suite.load_controller_config(default_controller=controller),
        horizon=200
    )

    obs = env.reset()
    pick_place = PickAndPlacePrimitive(env)

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
    frames = [np.flip(obs[f'{front_cam}_image'].transpose(2, 0, 1), axis=1) for obs in observations]
    wandb.log({'video': wandb.Video(np.asarray(frames), format='mp4', fps=20)})

    frames = [np.flip(obs[f'{agent_cam}_image'].transpose(2, 0, 1), axis=1) for obs in observations]
    wandb.log({'video': wandb.Video(np.asarray(frames), format='mp4', fps=20)})
        
    wandb.finish()