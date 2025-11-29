from dataclasses import dataclass, field
import os


EXPERIMENT_DIR = os.path.dirname(__file__)
CACHE_DIR = os.path.join(EXPERIMENT_DIR, "cache")
EXPERIMENT_NAME = os.path.basename(EXPERIMENT_DIR)
os.makedirs(CACHE_DIR, exist_ok=True)


@dataclass
class ModelConfig:
    """Configuration for the neural network architecture.

    Defines the UNet structure used for the diffusion model, including input dimensions,
    filter counts for encoder/decoder paths, and embedding dimensions.
    """
    input_size: tuple = (5, 96, 192)  # (channels, nlat, nlon)
    nside: int = 64  # HEALPix nside parameter
    enc_filters: tuple = (64, 128, 256, 512, 1024)  # Filter counts for each encoder block
    dec_filters: tuple = (512, 256, 128, 64, 64)  # Filter counts for each decoder block
    out_channels: int = 4  # Number of output channels
    temb_dim: int = 256  # Dimension for time embeddings
    doyemb_dim: int = 16  # Dimension for day-of-year embeddings
    healpix_emb_dim: int = 5  # Dimension for HEALPix embeddings
    context_channels: int = 1  # Number of context channels
    edges_path: str = os.path.join(CACHE_DIR, "edges.npz")  # Path to save/load HEALPix edges


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing.

    Specifies dataset paths, climate model, experiments, and pattern scaling parameters.
    """
    root_dir: str = "/orcd/data/raffaele/001/shahineb/products/cmip6/processed"  # CMIP6 data directory
    model_name: str = "MPI-ESM1-2-LR"  # Climate model to use
    train_experiments: dict = field(
        default_factory=lambda: {
            "piControl": ["r1i1p1f1"]
            # "historical": ["r1i1p1f1"],
            # "ssp126": [f"r{i + 1}i1p1f1" for i in range(3)],
            # "ssp585": [f"r{i + 1}i1p1f1" for i in range(3)]
        }
    )
    variables: tuple = ("tas", "pr", "hurs", "sfcWind")  # Climate variables
    train_gmst_path: str = os.path.join(CACHE_DIR, "gmsttrain.nc")  # Path to save/load precomputed annual GMST anomaly time series
    pattern_scaling_path: str = os.path.join(CACHE_DIR, "β.npy")  # Path to save/load pattern scaling coefficients
    norm_stats_path: str = os.path.join(CACHE_DIR, "μ_σ.npz")  # Path to save/load normalization statistics
    in_memory: bool = True  # Whether to load full dataset into memory
    norm_max_samples: int = 10000  # Maximum number of samples to use for normalization
    sigma_max_path: str = os.path.join(CACHE_DIR, "σmax.npy")  # Path to save/load σmax
    sigma_max_search_interval: tuple = (0, 200)  # Interval in which we search for sigma max


@dataclass
class TrainingConfig:
    """Configuration for model training.

    Defines hyperparameters, logging intervals, and output paths.
    """
    batch_size: int = 32  # Number of samples per batch
    learning_rate: float = 1e-4  # Adam optimizer learning rate
    ema_decay: float = 0.999  # Exponential moving average decay
    epochs: int = 2  # Number of training epochs
    log_interval: int = 20  # Steps between metric logging
    queue_length: int = 30  # Length of sliding window for metrics
    sample_interval: int = 1000  # Steps between sample generation
    sample_steps: int = 30  # Number of diffusion steps for sampling
    sample_count: int = 2  # Number of samples to generate
    random_seed: int = 0  # Seed for reproducibility
    checkpoint_interval: int = 1  # Epochs between checkpoints
    checkpoint_filename: str = os.path.join(CACHE_DIR, "ckpt.eqx")  # Output checkpoint filename
    model_filename: str = os.path.join(CACHE_DIR, "weights.eqx")  # Output model filename
    wandb_project: str = EXPERIMENT_NAME  # Weights & Biases project name


@dataclass
class ScheduleConfig:
    """Configuration for the diffusion process.

    Defines the noise schedule parameters for the variance exploding schedule.
    """
    sigma_max: float = 200   # Maximum noise level, if None then estimated from training data
    sigma_min: float = 1e-2  # Minimum noise level


@dataclass
class SamplingConfig:
    """Configuration for sampling and evaluation.

    Defines parameters for generating samples and computing metrics.
    """
    n_steps: int = 30  # Number of diffusion steps for sampling
    n_samples: int = 50  # Number of samples to generate per test point
    batch_size: int = 2  # Batch size for evaluation
    random_seed: int = 2100  # Seed for reproducibility
    output_dir: str = f"/orcd/data/raffaele/001/shahineb/emulated/climemu/experiments/daily/{EXPERIMENT_NAME}/outputs"  # Output directory for inference

@dataclass
class Config:
    """Master configuration that combines all sub-configurations."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
