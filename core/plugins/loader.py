"""
Plugin loader for Trove.

Discovers and loads plugins from a plugins/ directory. Each plugin
is a Python file or package that exposes a register(registry) function.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Optional

from core.plugins.registry import PluginRegistry

logger = logging.getLogger(__name__)


def discover_plugins(plugins_dir: Path) -> list[Path]:
    """Find all plugin modules in a directory.

    Discovers:
    - Single .py files (e.g. plugins/my_embedder.py)
    - Directories with __init__.py (e.g. plugins/my_embedder/)

    Ignores files starting with _ (e.g. __init__.py, __pycache__).

    Returns:
        Sorted list of plugin paths (files or directories).
    """
    if not plugins_dir.is_dir():
        return []

    plugins = []

    for entry in sorted(plugins_dir.iterdir()):
        if entry.name.startswith("_"):
            continue
        if entry.name.startswith("."):
            continue

        if entry.is_file() and entry.suffix == ".py":
            plugins.append(entry)
        elif entry.is_dir() and (entry / "__init__.py").exists():
            plugins.append(entry)

    return plugins


def load_plugin(path: Path, registry: PluginRegistry) -> bool:
    """Load a single plugin and call its register() function.

    Args:
        path: Path to plugin file (.py) or directory (with __init__.py).
        registry: The plugin registry to register into.

    Returns:
        True if plugin loaded successfully, False otherwise.
    """
    if path.is_file():
        module_name = f"trove_plugin_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
    elif path.is_dir():
        module_name = f"trove_plugin_{path.name}"
        init_path = path / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            module_name, init_path,
            submodule_search_locations=[str(path)],
        )
    else:
        logger.warning("Skipping invalid plugin path: %s", path)
        return False

    if spec is None or spec.loader is None:
        logger.warning("Could not create module spec for plugin: %s", path)
        return False

    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception:
        logger.exception("Failed to import plugin: %s", path)
        # Clean up partial import
        sys.modules.pop(module_name, None)
        return False

    register_fn = getattr(module, "register", None)
    if register_fn is None:
        logger.warning(
            "Plugin %s has no register() function, skipping", path.name
        )
        # Still keep it in sys.modules — it imported fine, just has no register
        return False

    try:
        register_fn(registry)
        logger.info("Loaded plugin: %s", path.name)
        return True
    except Exception:
        logger.exception("Plugin %s register() failed", path.name)
        return False


def load_plugins(
    plugins_dir: Optional[Path] = None,
    registry: Optional[PluginRegistry] = None,
) -> PluginRegistry:
    """Discover and load all plugins from a directory.

    Args:
        plugins_dir: Path to plugins directory. Defaults to ./plugins/
        registry: Existing registry to load into. Creates new one if None.

    Returns:
        The PluginRegistry with all successfully loaded plugins.
    """
    if registry is None:
        registry = PluginRegistry()

    if plugins_dir is None:
        plugins_dir = Path("plugins")

    if not plugins_dir.is_dir():
        logger.debug("No plugins directory at %s", plugins_dir)
        return registry

    plugin_paths = discover_plugins(plugins_dir)
    if not plugin_paths:
        logger.debug("No plugins found in %s", plugins_dir)
        return registry

    loaded = 0
    failed = 0
    for path in plugin_paths:
        if load_plugin(path, registry):
            loaded += 1
        else:
            failed += 1

    logger.info(
        "Plugin loading complete: %d loaded, %d failed (from %s)",
        loaded, failed, plugins_dir,
    )
    return registry
