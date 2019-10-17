from .ctvae import CTVAE
from .ctvae_mi import CTVAE_mi
from .ctvae_info import CTVAE_info
from .ctvae_style import CTVAE_style


model_dict = {
    'ctvae' : CTVAE,
    'ctvae_mi' : CTVAE_mi,
    'ctvae_info' : CTVAE_info,
    'ctvae_style' : CTVAE_style
}


def get_model_class(model_name):
    if model_name in model_dict:
        return model_dict[model_name]
    else:
        raise NotImplementedError
