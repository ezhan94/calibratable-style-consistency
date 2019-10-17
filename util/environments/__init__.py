from .core import BaseEnvironment, generate_rollout
from .bball import BBallEnv


environment_dict = {
    'bball' : BBallEnv()
}


def load_environment(env_name):
	if env_name in environment_dict:
		return environment_dict[env_name]
	else:
		raise NotImplementedError
