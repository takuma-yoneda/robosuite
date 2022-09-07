"""
This file contains the base wrapper class for Mujoco environments.
Wrappers are useful for data collection and logging. Highly recommended.
"""

from robosuite.wrappers import Wrapper


class ActionWrapper(Wrapper):
    """
    Action wrapper

    Args:
        env (MujocoEnv): The environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        """Convert action here"""
        raise NotImplementedError()

    def step(self, action):
        """
        By default, run the normal environment step() function

        Args:
            action (np.array): action to take in environment

        Returns:
            4-tuple:

                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        return self.env.step(self.action(action))

    @property
    def action_spec(self):
        """
        By default, grabs the normal environment action_spec

        Returns:
            2-tuple:

                - (np.array) minimum (low) action values
                - (np.array) maximum (high) action values
        """
        return self.env.action_spec

    @property
    def action_dim(self):
        """
        By default, grabs the normal environment action_dim

        Returns:
            int: Action space dimension
        """
        return self.env.dof