import yaml
import os
from src.utils.config_schema import ProjectConfig


def load_config(path: str | None = None) -> ProjectConfig:
    """
    Loads and validates project configuration using Pydantic.
    Converts all relative paths in the YAML into absolute paths based on the project root.
    """
    here = os.path.abspath(os.path.dirname(__file__))

    project_root = os.path.abspath(os.path.join(here, "..", ".."))

    candidates = []

    if path:
        candidates.append(path)

    candidates.append(os.path.join(project_root, "config", "local.yaml"))
    candidates.append(os.path.join(project_root, "..", "config", "local.yaml"))

    for candidate in candidates:
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

                abs_paths = {}
                for key, value in data["paths"].items():
                    if os.path.isabs(value):
                        abs_paths[key] = value
                    else:
                        abs_paths[key] = os.path.abspath(os.path.join(project_root, value))
                data["paths"] = abs_paths

                return ProjectConfig(**data)

    raise FileNotFoundError(f"No config file found. Checked: {candidates}")
