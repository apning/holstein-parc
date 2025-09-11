import argparse

from src.data_utils import unpickle_data
from src.get_settings import get_config
from src.training import train
from src.utils import str2bool

"""
Command-line interface for training models in Holstein-PARC.

Usage examples:
- Start a new run from a named config:
  python -m train --config_name="L_32-quench_0_1-steps_1000-training/medium_parc_model""

- Resume from a saved config, optionally applying additional named modifiers:
  python -m train --resume_path ".../path/to/LAST_CONFIG.pkl" --config_name "<modifier>"
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Holstein PARC models. Provide either a named config, a resume path, or both (to modify a resumed config).",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help=(
            "Slash-separated string specifying the base config and optional modifiers. "
            "Example: '1-quench-1a/c_128/dropout_0p2'. When used with --resume_path, these modifiers are applied on top of the resumed config."
        ),
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default=None,
        help=(
            "Path to a pickled Config object (typically '.../LAST_CONFIG.pkl') to resume from. "
            "If provided with --config_name, the named config will modify the resumed config before training."
        ),
    )
    parser.add_argument(
        "--suppress_tqdm",
        nargs="?",
        const=True,  # bare flag → True
        default=None,  # flag absent → None
        type=str2bool,  # explicit value parsed via helper
        help=(
            "If provided as a bare flag, disables tqdm progress bars (True). "
            "You can also pass explicit values: true/false, 1/0, yes/no, y/n. If omitted, existing config value is used."
        ),
    )

    args = parser.parse_args()
    config_name = args.config_name
    resume_path = args.resume_path
    suppress_tqdm = args.suppress_tqdm

    if config_name == "":
        config_name = None
    if resume_path == "":
        resume_path = None

    if config_name is None and resume_path is None:
        raise ValueError(
            "You must provide either a config_name and/or a resume_path. "
            "If you want to resume training, provide the resume_path. "
            "If you provide a resume_path and a config_name, the config_name will be used to modify the resuming config before training."
        )

    if resume_path is not None:
        config = unpickle_data(resume_path)
    else:
        config = None

    if config_name is not None:
        config = get_config(name=config_name, config=config, combine_subname=True)

    if suppress_tqdm is not None:
        config.suppress_tqdm = suppress_tqdm

    train(config=config)
