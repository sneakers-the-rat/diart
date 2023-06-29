import pytest

import numpy as np

@pytest.fixture
def config_args() -> dict:
    """
    Make random configuration parameters for the pipelineconfig
    """
    fixed = {
        'segmentation': 'pyannote/segmentation',
        'embedding': 'pyannote/embedding',
        'device': None,
        'hf_token': True
    }

    floats = {
        'step': np.random.rand(),
        'latency': np.random.rand(),
        'tau_active': np.random.rand(),
        'rho_update': np.random.rand(),
        'delta_new': np.random.rand(),
        'gamma': np.random.rand(),
        'beta': np.random.rand(),
        'duration': np.random.rand(),
    }

    ints = {
        'max_speakers': np.random.randint(1, 10),
    }

    config_dict = {**fixed, **floats, **ints}
    return config_dict



