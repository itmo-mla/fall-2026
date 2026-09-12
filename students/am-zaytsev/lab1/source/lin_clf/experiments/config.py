from dataclasses import dataclass


@dataclass
class TrainConfig:
    epochs: int = 10_000
    h: float | None = 0.001
    tao: float = 0.01
    momentum_k: float = 0.01
    batch_size: int = 2**6
    visual_smooth: float = 0.01
    seed: int = 42
    draw_loss: bool = True
    train_name: str = ""
    fetch_prob: bool = False
