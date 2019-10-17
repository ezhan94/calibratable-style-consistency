from util.environments import BaseEnvironment


class BBallEnv(BaseEnvironment):
    
    def __init__(self):
        pass

    def reset(self, init_state=[0,0]):
        # TODO change init state?
        self.state = init_state

    def get_obs(self):
        return self.state

    def step(self, action):
        self.state += action
        return self.state, 0.0, False, None
