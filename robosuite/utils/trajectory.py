class Trajectory:
    def __init__(self):
        self.clear()

    def add_transition(self, obs, act, rew, done, info):
        self.observations.append(obs)
        self.actions.append(act)
        self.rewards.append(rew)
        self.dones.append(done)
        self.infos.append(info)

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.infos = []

    def is_empty(self):
        if len(self.observations) == 0:
            return True
        return False

    @classmethod
    def from_transitions(cls, observations, actions, rewards, dones, infos):
        traj = cls()
        traj.observations = observations
        traj.actions = actions
        traj.rewards = rewards
        traj.dones = dones
        traj.infos = infos
        return traj

    def __add__(self, trajectory):
        obs = self.observations + trajectory.observations
        act = self.actions + trajectory.actions
        rew = self.rewards + trajectory.rewards
        dones = self.dones + trajectory.dones
        infos = self.infos + trajectory.infos
        return Trajectory.from_transitions(obs, act, rew, dones, infos)