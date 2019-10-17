import torch


class BaseEnvironment(object):

    def __init__(self):
        raise NotImplementedError

    def reset(self):
        print('default reset')
        raise NotImplementedError

    def get_obs(self):
        return NotImplementedError

    def step(self, action):
        raise NotImplementedError  


def generate_rollout(env, model, burn_in=0, burn_in_actions=None, horizon=0):
    if burn_in > 0:
        assert burn_in_actions is not None

    state = env.get_obs() # should be reset?

    states = [state.clone().unsqueeze(0)]
    actions = []

    for t in range(horizon):
        if t < burn_in:
            action = burn_in_actions[t]
            if model.is_recurrent:
                model.update_hidden(state, action)
        else:
            action = model.act(state)

        obs, reward, done, info = env.step(action)
        assert not done

        states.append(state.clone().unsqueeze(0))
        actions.append(action.unsqueeze(0))

    return torch.cat(states, dim=0), torch.cat(actions, dim=0)
