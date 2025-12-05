"""Path utilities for accessing package resources"""

from pathlib import Path
from typing import Optional

try:
    # Python 3.9+
    from importlib.resources import files
except ImportError:
    # Python 3.8
    from importlib_resources import files


def get_package_root() -> Path:
    """
    Get the root directory of the package

    Returns:
        Path to package root (src directory)
    """
    return Path(__file__).parent.parent


def get_config_dir() -> Path:
    """
    Get the config directory path

    Returns:
        Path to config directory
    """
    return get_package_root() / "config"


def get_config_path(config_name: str) -> Path:
    """
    Get path to a specific config file

    Args:
        config_name: Config file name (e.g., "default.yaml", "tiny_fast")

    Returns:
        Path to config file

    Examples:
        >>> get_config_path("default.yaml")
        >>> get_config_path("tiny_fast")  # .yaml is auto-added
    """
    # Add .yaml extension if not present
    if not config_name.endswith('.yaml'):
        config_name = f"{config_name}.yaml"

    config_path = get_config_dir() / config_name

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_name}\n"
            f"Available configs: {list_available_configs()}"
        )

    return config_path


def list_available_configs() -> list[str]:
    """
    List all available config files

    Returns:
        List of config file names (without .yaml extension)
    """
    config_dir = get_config_dir()

    if not config_dir.exists():
        return []

    configs = [
        f.stem for f in config_dir.glob("*.yaml")
    ]

    return sorted(configs)


def resolve_config_path(config_path: Optional[str] = None) -> Path:
    """
    Resolve config path with fallback to default

    Args:
        config_path: Optional config path or name

    Returns:
        Resolved Path object

    Examples:
        >>> resolve_config_path()  # Returns default.yaml
        >>> resolve_config_path("tiny_fast")  # Returns tiny_fast.yaml
        >>> resolve_config_path("custom.yaml")  # Returns custom.yaml if exists
        >>> resolve_config_path("/abs/path/config.yaml")  # Returns absolute path
    """
    if config_path is None:
        # Use default config
        return get_config_path("default.yaml")

    config_path_obj = Path(config_path)

    # If absolute or relative path to existing file
    if config_path_obj.exists():
        return config_path_obj

    # Try to find in package configs
    try:
        return get_config_path(config_path)
    except FileNotFoundError:
        # If not found in package, return as-is (will fail in load_config)
        return config_path_obj
