import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Type, TypeVar

from omegaconf import MISSING, DictConfig, OmegaConf

# Type variable for structured configs
T = TypeVar("T")


def load_config(
    config_path: str, cli_overrides: Optional[list[str]] = None
) -> DictConfig:
    """Load configuration from YAML file and apply CLI overrides.

    Args:
        config_path: Path to the base YAML configuration file
        cli_overrides: List of dotlist overrides (e.g., ["learning_rate=0.01", "batch_size=64"])

    Returns:
        Merged OmegaConf configuration
    """
    # Load base config
    cfg = OmegaConf.load(config_path)

    # Merge with CLI overrides (dotlist format)
    if cli_overrides:
        cli_cfg = OmegaConf.from_dotlist(cli_overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    return cfg


def save_config(cfg: DictConfig, exp_folder: str) -> None:
    """Save configuration to experiment folder.

    Args:
        cfg: OmegaConf configuration to save
        exp_folder: Experiment directory path
    """
    Path(exp_folder).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, f"{exp_folder}/config.yaml")


def merge_with_defaults(
    defaults: DictConfig,
    config_path: Optional[str] = None,
    overrides: Optional[list[str]] = None,
) -> DictConfig:
    """Merge defaults with optional config file and CLI overrides.

    Args:
        defaults: Default configuration as DictConfig
        config_path: Optional path to YAML config file
        overrides: Optional list of dotlist overrides

    Returns:
        Merged configuration
    """
    cfg = OmegaConf.create(defaults)

    # Merge with config file if provided
    if config_path and os.path.exists(config_path):
        file_cfg = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(cfg, file_cfg)

    # Apply CLI overrides (highest priority)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    return cfg


def load_structured_config(
    config_class: Type[T],
    config_path: Optional[str] = None,
    overrides: Optional[list[str]] = None,
) -> T:
    """Load structured config from dataclass with optional file and CLI overrides.

    This function creates a structured config with type safety and validation:
    - Required fields (marked with MISSING) must be provided via config file or overrides
    - Type validation is performed automatically
    - Better IDE support with autocomplete

    Args:
        config_class: Dataclass type for structured config
        config_path: Optional path to YAML config file
        overrides: Optional list of dotlist overrides

    Returns:
        Structured configuration instance

    Raises:
        omegaconf.errors.MissingMandatoryValue: If required fields are not provided

    Example:
        >>> cfg = load_structured_config(
        ...     MOTGraphTrainingConfig,
        ...     config_path="configs/mot-gnn.yaml",
        ...     overrides=["lr=0.001", "batch_size=64"]
        ... )
    """
    # Create structured config from dataclass
    cfg = OmegaConf.structured(config_class)

    # Merge with config file if provided
    if config_path and os.path.exists(config_path):
        file_cfg = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(cfg, file_cfg)

    # Apply CLI overrides (highest priority)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    # Convert back to structured config (validates required fields)
    return OmegaConf.to_object(cfg)


def merge_with_args_section(
    defaults: DictConfig,
    config_path: str,
    overrides: Optional[list[str]] = None,
) -> DictConfig:
    """Merge defaults with config file's 'args' section and CLI overrides.

    This is specific to the RL training workflow where conf.yaml has an 'args'
    section that overrides default training arguments.

    Args:
        defaults: Default configuration as DictConfig
        config_path: Path to YAML config file (e.g., conf.yaml in logdir)
        overrides: Optional list of dotlist overrides

    Returns:
        Merged configuration
    """
    raise NotImplementedError
    cfg = OmegaConf.create(defaults)

    # Load conf.yaml
    if os.path.exists(config_path):
        conf_yaml = OmegaConf.load(config_path)

        # Merge with args section if present
        if "args" in conf_yaml:
            cfg = OmegaConf.merge(cfg, conf_yaml.args)

    # Apply CLI overrides (highest priority)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    return cfg


def resolve_logdir(logdir: Optional[str]) -> str:
    """Resolve and validate logdir path.

    Args:
        logdir: Log directory path

    Returns:
        Absolute path to logdir

    Raises:
        ValueError: If logdir is not provided or doesn't exist
    """
    if not logdir:
        raise ValueError("logdir must be provided")

    logdir_path = os.path.realpath(logdir)

    if not os.path.isdir(logdir_path):
        raise ValueError(f"Log directory does not exist: {logdir_path}")

    return logdir_path


def print_config(cfg: DictConfig, title: str = "Configuration") -> None:
    """Pretty print configuration.

    Args:
        cfg: OmegaConf configuration to print
        title: Title to display
    """
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    print(OmegaConf.to_yaml(cfg))
    print(f"{'='*70}\n")


@dataclass
class MOTGraphTrainingConfig:
    """Configuration for MOT Graph (GNN) training.

    Required parameters are marked with MISSING and must be provided
    via config file or CLI overrides.
    """

    # ===== Required Parameters (MISSING) =====
    output_dir: str = MISSING  # Output directory (required)

    # ===== Data Arguments =====
    mot_dirs: list[str] = field(
        default_factory=lambda: [
            "data-sync/MOT17/train-unified/MOT17-02",
            "data-sync/MOT17/train-unified/MOT17-04",
            "data-sync/MOT17/train-unified/MOT17-05",
            "data-sync/MOT17/train-unified/MOT17-09",
            "data-sync/MOT17/train-unified/MOT17-10",
            "data-sync/MOT17/train-unified/MOT17-11",
            "data-sync/MOT17/train-unified/MOT17-13",
        ]
    )
    ann_filename: Optional[str] = None  # Annotation filename (None = GT)
    use_precomputed_features: bool = False  # Use precomputed RPN features
    feature_filename: Optional[str] = None  # Feature file path
    no_bar: bool = False  # Disable progress bars
    max_samples: Optional[int] = None  # Max samples to load

    # ===== Model Arguments =====
    task_type: str = "classification"  # Task type: regression or classification

    # ===== Training Arguments =====
    device: str = "cuda:0"  # Device for training
    epochs: int = 20  # Number of training epochs
    lr: float = 0.01  # Learning rate
    batch_size: int = 32  # Batch size
    num_samples: int = 2000  # Samples per epoch
    grad_clip: float = 1.0  # Gradient clipping threshold
    log_interval: int = 50  # Logging interval (batches)

    # ===== Dataset Arguments =====
    agent_radius: float = 0.02  # Agent radius
    max_entities: int = 1000  # Max entities per sample
    val_int: int = 10  # Validation interval (epochs)
    num_workers: int = 4  # DataLoader workers
    connect_threshold: float = 50.0  # Edge connection threshold
    image_size: list[int] = field(default_factory=lambda: [500, 500])  # Image size
    include_agent_node: bool = False  # Include agent as graph node
    image_cache_size: int = 2000  # Image cache size (LRU cache for faster loading)

    # ===== System =====
    benchmark: bool = False  # Enable timing benchmarks

    # ===== Experiment Settings =====
    # There are a few more, commented out for now
    run: Optional[str] = None
    base: Optional[Dict] = None
    tune: Optional[Dict] = None
    # TODO: eval should be a subclass or something


@dataclass
class MOTGraphEvaluationConfig:
    """Configuration for MOT Graph model evaluation.

    Required parameters are marked with MISSING and must be provided
    via config file or CLI overrides.
    """

    # ===== Required Parameters (MISSING) =====
    checkpoint_dir: str = MISSING  # Checkpoint directory (required)
    mot_dirs: list[str] = MISSING  # MOT data directories (required)

    # ===== Data Arguments =====
    ann_filename: Optional[str] = None  # Annotation filename (None = GT)
    use_precomputed_features: bool = False  # Use precomputed RPN features
    feature_filename: str = "features.npz"  # Feature file path
    no_bar: bool = False  # Disable progress bar
    max_samples: Optional[int] = None  # Max samples to load

    # ===== Evaluation Arguments =====
    device: str = "cuda:0"  # Device for evaluation
    batch_size: int = 32  # Batch size
    num_workers: int = 4  # DataLoader workers
    num_samples: int = 5000  # Number of samples to evaluate

    # ===== Dataset Parameters =====
    agent_radius: float = 0.02  # Agent radius
    max_entities: int = 100  # Max entities per sample
    connect_threshold: float = 50.0  # Edge connection threshold
    image_size: list[int] = field(default_factory=lambda: [500, 500])  # Image size
    include_agent_node: bool = False  # Include agent node in graph
    task_type: str = "regression"  # Task type: regression or classification

    # ===== Detection Metrics =====
    compute_detection_metrics: bool = False  # Compute detection metrics
    max_detection_batches: int = 5  # Max batches for detection metrics
    iou_threshold: float = 0.5  # IoU threshold for detection

    # ===== Output =====
    output: Optional[str] = None  # Path to save results JSON
    benchmark: bool = False  # Enable timing benchmarks


@dataclass
class MOTInferenceConfig:
    """Configuration for MOT inference and visualization.

    Required parameters are marked with MISSING and must be provided
    via config file or CLI overrides.
    """

    # ===== Required Parameters (MISSING) =====
    checkpoint_dir: str = MISSING  # Checkpoint directory (required)
    mot_dir: str = MISSING  # MOT data directory (required)
    output_dir: str = MISSING  # Output directory for videos (required)

    # ===== Data Arguments =====
    sequence: Optional[str] = None  # Specific sequence (None = all in mot_dir)
    max_frames: Optional[int] = None  # Max frames to process (None = all)
    start_frame: int = 1  # Starting frame number
    ann_filename: Optional[str] = None  # Annotation filename (None = GT)

    # ===== Model Arguments =====
    use_precomputed_features: bool = False  # Use precomputed RPN features
    feature_filename: Optional[str] = None  # Feature file path
    task_type: str = "regression"  # Task type: regression or classification

    # ===== Visualization Arguments =====
    fps: int = 25  # Output video FPS
    show_agent: bool = True  # Show agent position
    show_value: bool = True  # Show value predictions
    show_edges: bool = False  # Show graph edges
    scale: float = 1.0  # Visualization scale factor
    no_video: bool = False  # Skip video output, only process
    show_gt: bool = True  # Show ground truth boxes

    # ===== Dataset Parameters =====
    agent_radius: float = 0.02  # Agent radius
    max_entities: int = 100  # Max entities per sample
    connect_threshold: float = 50.0  # Edge connection threshold
    image_size: list[int] = field(default_factory=lambda: [500, 500])  # Image size
    include_agent_node: bool = False  # Include agent as graph node

    # ===== System =====
    device: str = "cuda:0"  # Device for inference
    batch_size: int = 1  # Batch size (usually 1 for inference)
    num_workers: int = 0  # DataLoader workers (0 for inference)
    seed: Optional[int] = None  # Random seed for reproducibility


@dataclass
class RLTrainingConfig:
    """Configuration for RL training with Ray RLlib.

    Required parameters are marked with MISSING and must be provided
    via config file or CLI overrides.
    """

    # ===== Required Parameters (MISSING) =====
    logdir: str = MISSING  # Log directory containing conf.yaml (required)

    # ===== General Settings =====
    verbose: bool = False  # Enable verbose logging
    smoke: bool = False  # Run smoke test (2 iterations)
    resume_from: Optional[str] = None  # Checkpoint path to resume from
    errored_only: bool = False  # Only resume errored trials
    tune: bool = False  # Enable Ray Tune hyperparameter tuning
    no_sched: bool = False  # Disable scheduler

    # ===== Stopping Criteria =====
    stop_attr: str = "timesteps_total"  # Stopping attribute
    stop_at: int = 100000  # Stopping value
    num_samples: int = 1  # Number of samples for tuning

    # ===== Resource Allocation =====
    local: bool = False  # Run in local mode
    concurrency: int = 100000  # Max concurrent trials
    num_workers: int = 1  # Number of rollout workers
    cpus_per_worker: float = 1.0  # CPUs per worker
    gpus_per_worker: float = 0.0  # GPUs per worker
    envs_per_worker: int = 1  # Environments per worker
    grace_period: float = 0.25  # Grace period for scheduler
    num_gpus: float = 1.0  # Total GPUs for training
    worker_gpu: bool = False  # Allocate GPUs to workers
    no_gpu_workers: bool = False  # Custom resource for no-GPU workers

    # ===== Training Settings =====
    amp: bool = False  # Enable automatic mixed precision
    show_model: bool = False  # Show model architecture
    checkpoint_freq: int = 1  # Checkpoint frequency
    keep_all_chkp: bool = False  # Keep all checkpoints
    eval_int: Optional[int] = None  # Evaluation interval

    # ===== Cluster Settings =====
    node_ip: str = "127.0.0.1"  # Node IP address
    head_ip: Optional[str] = None  # Ray head node IP
    num_cpus: Optional[str] = None  # Number of CPUs

    # ===== Experiment Settings =====
    exp_name: str = "SPG-EXP"  # Experiment name
    callbacks: Optional[list[str]] = None  # Callback list


OmegaConf.structured(MOTGraphTrainingConfig)
