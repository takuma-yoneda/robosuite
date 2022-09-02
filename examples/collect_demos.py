"""Run expert policy on block-insertion on robosuite, and collect trajectories, store them in diffuser dataset format"""

import os
import torch
from pathlib import Path
import numpy as np
from robosuite.environments.manipulation.wrappers import GripperAbsRotWrapper
# import diffuser.datasets as datasets
from robosuite.primitives import PickAndPlaceAbsPrimitive
from robosuite.utils.trajectory import Trajectory
import robosuite as suite
import wandb
from scipy.spatial.transform import Rotation as R


def add_text(img, text, **kwargs):
    import numpy as np
    import cv2
    _kwargs = {
        'org': (200, 200),
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

def decorate_obs(img, eef_pos, eef_ori, act=None):
    img = add_text(img, 'eef_pos: ' + ' '.join([f'{entry:.4f}' for entry in eef_pos.tolist()]), org=(0, 20))
    img = add_text(img, 'eef_ori: ' + ' '.join([f'{entry:.4f}' for entry in eef_ori.tolist()]), org=(0, 40))
    if act is not None:
        img = add_text(img, 'act: ' + ' '.join([f'{entry:.4f}' for entry in act]), org=(0, 60))
    return img

def get_frame(obs, act=None):
    img = np.flip(obs['agentview_image'], axis=1)  # Is this necessary??
    img = np.flip(img, axis=0)
    eef_pos = obs['robot0_eef_pos']
    eef_ori = R.from_quat(obs['robot0_eef_quat']).as_euler('xyz')

    img = decorate_obs(img, eef_pos, eef_ori, act)
    return img.transpose(2, 0, 1)

gripper_ori = np.array([-np.pi, 0, 0])
def rollout_trajectory(env):
    obs = env.reset()
    pick_place = PickAndPlaceAbsPrimitive(env, init_obs=obs, pregrasp_height=0.2, interpolate=True)
    pick_pos, pick_quat = obs['blockA_pos'], obs['blockA_quat']
    place_pos = obs['goal_pos']
    cube_halfsize = obs['blockA_pos'][2] - pick_place.table_height
    block_ori = R.from_quat(pick_quat).as_euler('xyz')
    pick_ori = np.array([gripper_ori[0], gripper_ori[1], block_ori[2] - np.pi / 2])
    pick_pos[2] = pick_place.table_height + 0.02
    place_ori = np.array([gripper_ori[0], gripper_ori[1], - np.pi / 2])
    place_pos[2] += cube_halfsize

    pick_pose = np.array([*pick_pos, *pick_ori])
    place_pose = np.array([*place_pos, *place_ori])

    pick_trajectory, grasp_success = pick_place.pick(pick_pose, env.blockA)
    print('grasp success', grasp_success)
    print('grasp success (obs)', [obs['gripper_collision'] for obs in pick_trajectory.observations])
    if not pick_trajectory.dones[-1]:
        place_trajectory = pick_place.place(place_pose)
    else:
        place_trajectory = Trajectory()

    # TODO: Check if successful

    return pick_trajectory + place_trajectory


def main(env_name, gripper_types, control_freq, horizon, num_episodes):
    save_dir = Path(os.getenv('RMX_MOUNT_DIR')) / 'trajectories' / env_name
    save_dir.mkdir(exist_ok=True, parents=True)
    controller_config = suite.load_controller_config(default_controller='OSC_POSE_ABS')

    # cam_names=['frontview', 'agentview', 'leftview', 'rightview']
    cam_names=['agentview']
    env = suite.make(
        env_name=env_name,
        robots="UR5e",
        gripper_types=gripper_types,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        # camera_names=[front_cam, agent_cam],
        # camera_names=['frontview', 'agentview', 'leftview', 'rightview'],
        camera_names=cam_names,
        camera_depths=True,
        camera_heights=480,
        camera_widths=640,
        control_freq=control_freq,
        controller_configs=controller_config,
        horizon=horizon
    )

    env = GripperAbsRotWrapper(env)

    idx = 0
    while idx < num_episodes:
        trajectory = rollout_trajectory(env)
        sum_rewards = sum(trajectory.rewards)
        print(f'ep_len: {len(trajectory.observations)}\tsum_rewards: {sum_rewards}')
        if sum_rewards == 0:
            print('sum_rewards is zero!! trying again.')
            continue

        # NOTE: Why I didn't have this before and still had no issues???
        trajectory.dones[-1] = True

        torch.save(trajectory, save_dir / f'traj_{idx:04d}.pkl')

        # TODO: save the trajectory

        # Save video
        # frames = [np.flip(obs[f'agentview_image'].transpose(2, 0, 1), axis=1) for obs in trajectory.observations]
        frames = [get_frame(obs, act) for obs, act in zip(trajectory.observations, trajectory.actions)]
        # left_frames = [np.flip(obs[f'leftview_image'].transpose(2, 0, 1), axis=1) for obs in trajectory.observations]
        # right_frames = [np.flip(obs[f'rightview_image'].transpose(2, 0, 1), axis=1) for obs in trajectory.observations]
        wandb.log({
            'agentview': wandb.Video(np.asarray(frames), format='mp4', fps=4),
            # 'leftview': wandb.Video(np.asarray(left_frames), format='mp4', fps=4),
            # 'rightview': wandb.Video(np.asarray(right_frames), format='mp4', fps=4),
            'step': idx,
            'sum_rewards': sum_rewards
        })
        idx += 1

if __name__ == '__main__':
    config = dict(
        num_episodes=200,
        horizon=50,
        env_name="PickPlace",
        gripper_types="RobotiqThreeFingerAbsoluteGripper",
        control_freq=1
    )

    wandb.login()  # NOTE: You need to set envvar WANDB_API_KEY
    wandb.init(
        # Set the project where this run will be logged
        project='robosuite-test',
        # Track hyperparameters and run metadata
        config=config
        # config=ars(Args),
        # resume=True,
        # id=wandb_runid
    )
    main(**config)

    wandb.finish()

