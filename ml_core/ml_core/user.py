from dataclasses import dataclass


@dataclass
class User:
    mlruns_dir: str
    device: str = "cpu"
    random_seed: int = 42
