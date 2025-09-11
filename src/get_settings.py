import sys
from src.config import Config
from src.utils import get_project_root

# Add settings folder directly to Python path for imports
settings_path = get_project_root() / "settings"
if str(settings_path) not in sys.path:
    sys.path.insert(0, str(settings_path))

from cirriculums import CIRRICULUM_NAMED_KWARGS  # noqa: E402
from configs import CONFIG_NAMED_KWARGS  # noqa: E402


def get_cirriculum(cirriculum_name: str, noise_coeff: float | None = None) -> list[dict]:
    """
    Return a particular cirriculum by its name

    Args:
        cirriculum_name (str): The name of the cirriculum
        noise_coeff (float|None): The noise coefficient for the cirriculum, which modifies the noise values. If None, the original noise values will be used (None is the same as 1.0)

    """

    if cirriculum_name not in CIRRICULUM_NAMED_KWARGS:
        raise ValueError(
            f"Cirriculum {cirriculum_name} not found. Available cirriculums: {list(CIRRICULUM_NAMED_KWARGS.keys())}"
        )

    cirriculum = CIRRICULUM_NAMED_KWARGS[cirriculum_name]

    if noise_coeff is not None:
        for stage in cirriculum:
            stage["noise"] *= noise_coeff

    return cirriculum


def get_config(name: str, config: Config | None = None, combine_subname: bool = False) -> Config:
    """
    Return a particular Config object with kwargs specified by name.
    name is a string with name-parts separated with forward slashs "/".
    Ie. name = "{name_part_1}/{name_part_2}/{name_part_3}/..." (Note: Empty strings after the split will be discarded. So "/example///" is the same as "example")
    If Config object is not specified, name_part_1 will be used to create a new Config object. Subsequent name-parts will be used to modify the object in the order provided. If a Config object is provided, all name-parts will be used to modify it.

    if combine_subname is True, then the sub-names of all name parts will be combined, instead of replacing each other.

    Args:
        name (str): The name of the desired Config kwargs. See rules above

        config (Config | None): An existing config object. If specified, will modify this object instead of instantiating a new one

        combine_subname (bool): Whether to combine sub_name of name parts instead of them replacing each other.

    Returns:
        Config: The new/modified Config object
    """

    name_parts = [part for part in name.split("/") if part]

    if len(name_parts) == 0:
        raise ValueError(
            "Name must contain at least one part. Use a non-empty string with characters other than just '/'."
        )

    for name_part in name_parts:
        if config is None:
            config = Config(**CONFIG_NAMED_KWARGS[name_part])
        else:
            if name_part in CONFIG_NAMED_KWARGS:
                kwargs = CONFIG_NAMED_KWARGS[name_part]
            elif name_part in CIRRICULUM_NAMED_KWARGS:
                kwargs = {"cirriculum_name": name_part, "sub_name": name_part}
            else:
                raise ValueError(f"The name part '{name_part}' does not exist")

            config.update_attrs_with_dict(
                kwargs=kwargs, must_exist=True, combine_notes=True, combine_subname=combine_subname
            )

    # To define a history on how this config was created.
    # In theory _config_name should be able to mostly recreate the config, sans any outside modifications (such as modifications made by the training process)
    if not hasattr(config, "_config_name"):
        config._config_name = None
    if config._config_name is None:
        config._config_name = name
    else:
        config._config_name += "/" + name

    return config
