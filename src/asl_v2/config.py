from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathsConfig:
    dataset_root: Path
    artifacts_dir: Path = Path("artifacts")


@dataclass(frozen=True)
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42
    batch_size: int = 32
    epochs: int = 30
    patience: int = 5
