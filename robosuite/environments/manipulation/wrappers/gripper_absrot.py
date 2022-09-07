import numpy as np
from robosuite.wrappers import ActionWrapper
from scipy.spatial.transform import Rotation as R
import robosuite.utils.transform_utils as T
    
class GripperAbsRotWrapper(ActionWrapper):
    """Enables the user to specify absolute orientation of the gripper.

    When you use OSC_POSE controller with control_delta = False (i.e., absolute pose control),
    users need to specify gripper's position and orientation.
    As default, the input orientation is for "gripper0_grip_site" object 
    even though the observation for orientation ("robot0_eef_quat") is for another object "robot0_right_hand".
    This is really confusing. This wrapper enables the user to specify the orientation of "robot0_right_hand".
    """

    def __init__(self, env):
        super().__init__(env)
        assert env.robots[0].controller.use_ori, "use_ori must be True to use GripperAbsRotWrapper"
        assert not env.robots[0].controller.use_delta, "control_delta must be True to use GripperAbsRotWrapper"
        # Keep the rotation from "robot0_right_hand" to "gripper0_grip_site"
        Rwg = np.array(
            env.sim.data.site_xmat[env.sim.model.site_name2id(
                'gripper0_grip_site')].reshape([3, 3])
        )
        p_g = env.sim.data.get_site_xpos('gripper0_grip_site')

        Rwh = T.quat2mat(
            T.convert_quat(env.sim.data.get_body_xquat('robot0_right_hand'), to="xyzw")
        )
        p_h = env.sim.data.get_body_xpos('robot0_right_hand')

        # Construct transform from hand to gripper
        Rgh = Rwg.T @ Rwh
        Tgh = np.zeros((4, 4))
        Tgh[:3, :3] = Rgh
        Tgh[3, :3] = p_g - p_h
        Tgh[3, 3] = 1
        self.Tgh = Tgh

        Rhg = Rgh.T
        Thg = np.zeros((4, 4))
        Thg[:3, :3] = Rhg
        Thg[3, :3] = p_g - p_h
        Thg[3, 3] = 1
        self.Thg = Thg
    
    def action(self, act):
        # NOTE: orientation is specified as a rot vector (axis-angle representation)!!
        tgt_hand_pos = act[:3]
        tgt_hand_euler = act[3:6]

        gripper_ori_mat = np.array(
            self.env.sim.data.site_xmat[self.env.sim.model.site_name2id(
                'gripper0_grip_site')].reshape([3, 3])
        )
        gripper_pos = self.env.sim.data.get_site_xpos('gripper0_grip_site')

        gripper_ori = R.from_matrix(gripper_ori_mat).as_rotvec()
        
        # Calculate the corresponding pose of gripper in the world frame

        # (target) Transformation from world frame to hand frame
        Twh = np.zeros((4, 4))
        Twh[:3, :3] = R.from_euler('xyz', tgt_hand_euler).as_matrix()
        Twh[3, :3] = np.array(tgt_hand_pos)
        Twh[3, 3] = 1

        # Transformation from world frame to gripper frame
        Twg = Twh @ self.Thg

        # NOTE: I know it's crazy, but action space is
        # + pos --> p_h
        # + ori --> Rwg  (as rotvec)
        Rwg = Twg[:3, :3]
        rotvec = R.from_matrix(Rwg).as_rotvec()
        pos = Twh[3, :3]

        return np.array([*pos, *rotvec, *act[6:]])