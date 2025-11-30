import equinox as eqx

from src.diffusion import Song2020HealPIXUNet, ContinuousVESchedule
from experiments.mpisong.config import Config
from experiments.mpisong.data import load_dataset, compute_normalization, estimate_sigma_max
from experiments.mpisong.trainer import train
from experiments.mpisong.utils import load_or_compute_edges, print_parameter_count


def main():
    """Main entry point for training the climate diffusion model.
    
    This function:
    1. Loads configuration
    2. Prepares training and validation datasets
    3. Computes normalization statistics + maximum noise level
    4. Sets up the diffusion process and model
    5. Trains the model
    6. Saves the trained model
    """
    # Load configuration
    config = Config()

    # Load training dataset with pattern scaling
    train_dataset = load_dataset(
        root=config.data.root_dir,
        model=config.data.model_name,
        experiments=config.data.train_experiments,
        variables=config.data.variables,
        gmst_path=config.data.train_gmst_path,
        pattern_scaling_path=config.data.pattern_scaling_path
    )
    
    # Compute normalization statistics from a random subset of the training data
    μ_train, σ_train = compute_normalization(
        train_dataset,
        config.training.batch_size,
        max_samples=config.data.norm_max_samples,
        seed=config.training.random_seed,
        norm_stats_path=config.data.norm_stats_path
    )

    # Estimate sigma_max for the dataset (if not already cached)
    sigma_max = estimate_sigma_max(
        dataset=train_dataset,
        μ=μ_train,
        σ=σ_train,
        ctx_size=config.model.context_channels,
        search_interval=config.data.sigma_max_search_interval,
        seed=config.training.random_seed,
        sigma_max_path=config.data.sigma_max_path
    )

    # Setup noise schedule for the diffusion process
    if config.schedule.sigma_max:
        sigma_max = config.schedule.sigma_max
    schedule = ContinuousVESchedule(config.schedule.sigma_min, sigma_max)

    # Load or compute Latlon-HEALPix edges
    edges_to_healpix, edges_to_latlon = load_or_compute_edges(
        nside=config.model.nside,
        lat=train_dataset.cmip6data.lat,
        lon=train_dataset.cmip6data.lon,
        edges_path=config.model.edges_path
    )

    # Initialize the UNet model for diffusion
    model = Song2020HealPIXUNet(
        input_size=(5, 96, 192),
        nside=64,
        n_filters=[64, 128, 128, 128],
        n_blocks=[2, 2, 2, 2],
        out_channels=4,
        temb_dim=256,
        doyemb_dim=16,
        posemb_dim=64,
        healpix_emb_dim=5,
        n_bottleneck_blocks=1,
        edges_to_healpix=edges_to_healpix,
        edges_to_latlon=edges_to_latlon
    )
    print_parameter_count(model)
    
    # Train the model
    model = train(model, train_dataset, schedule, μ_train, σ_train, config)
    
    # Save the trained model
    eqx.tree_serialise_leaves(config.training.model_filename, model)
    print(f"Model saved to {config.training.model_filename}")


if __name__ == "__main__":
    main()