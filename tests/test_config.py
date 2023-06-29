import pytest

import numpy as np

from diart.blocks.config import PipelineConfig
from diart.models import SegmentationModel, EmbeddingModel
from diart.console.benchmark import parse_args

from fixtures import config_args

def test_from_dict(config_args):
    """
    Following the example in benchmark, user should be able to
    configure pipeline from a dictionary.

    Configuring with a dict should yield the same configuration as configuration
    using __init__
    """

    config = PipelineConfig.from_dict(config_args)
    config_2 = PipelineConfig(**config_args)

    # don't test for these by iteration since they get changed after instantiation
    exclude_params = (
        'segmentation',
        'embedding',
        'device',
        'hf_token'
    )

    for key, val in config_args.items():
        if key in exclude_params:
            continue
        assert getattr(config, key) == val
        assert getattr(config, key) == getattr(config_2, key)

    assert isinstance(config.segmentation, SegmentationModel)
    assert isinstance(config.embedding, EmbeddingModel)
    # not sure what device is supposed to be yet, or hf_token


def test_argparse_config(config_args):

    args = []
    for k, v in config_args.items():
        args.append(f'--{k}')
        args.append(str(v))

    cli_args = parse_args(args)

    test_from_dict(cli_args)




