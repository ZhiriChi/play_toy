from pathlib import Path
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for key in ("data_dir", "uploads_dir", "vector_dir"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["db_path"]).parent.mkdir(parents=True, exist_ok=True)
    return cfg
