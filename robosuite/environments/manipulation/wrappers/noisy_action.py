from robosuite.wrappers import ActionWrapper
import numpy as np
import random

class NoisyActionWrapper(ActionWrapper):
    def __init__(self, env, pos_scale=0.02, ori_scale=0.4, grip_scale=0.1, ori_z_only=True):
        super().__init__(env)
        assert self.action_dim == 3 + 3 + 1  # pos 3, ori 3, gripper 1
        self.pos_scale = pos_scale
        self.ori_scale = ori_scale
        self.grip_scale = grip_scale
        self.ori_z_only = ori_z_only

    def action(self, action):

        # Add some noise to the action
        pos_act, ori_act = action[:3], action[3:6]
        gripper_act = action[6]

        pos_noise = self.pos_scale * np.random.random((3,))
        ori_noise = self.ori_scale * np.random.random((3,))
        if self.ori_z_only:
            ori_noise[:2] = 0.
        gripper_noise = self.grip_scale * random.random()
        return [*(pos_act + pos_noise), *(ori_act + ori_noise), gripper_act + gripper_noise]