import os
import jax.numpy as jnp
import equinox as eqx

from src.diffusion import HealPIXUNet, ContinuousVESchedule
from .config import Config
from .data import load_dataset, compute_normalization, estimate_sigma_max
from .trainer import train
from .utils import load_or_compute_edges, print_parameter_count


class Denoiser(eqx.Module):
    unet: HealPIXUNet
    ctx_size: int = eqx.field(static=True)
    def __call__(self, x, σ):
        def c_skip(σ): return 1 / jnp.sqrt(1 + σ**2)
        def c_out(σ): return σ / jnp.sqrt(1 + σ**2)
        return c_skip(σ) * x[:-self.ctx_size] + c_out(σ) * self.unet(x, σ)


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
        in_memory=config.data.in_memory,
        pattern_scaling_path=config.data.pattern_scaling_path
    )
    
    # Load validation dataset using pattern scaling from training
    # This ensures consistent pattern scaling between train and validation
    val_dataset = load_dataset(
        root=config.data.root_dir,
        model=config.data.model_name,
        experiments=config.data.val_experiments,
        variables=config.data.variables,
        subset={'time': slice(*config.data.val_time_slice)},
        in_memory=config.data.in_memory,
        external_β=train_dataset.β  # Use training pattern scaling coefficients
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
    model = HealPIXUNet(
        input_size=config.model.input_size,
        nside=config.model.nside,
        enc_filters=list(config.model.enc_filters),
        dec_filters=list(config.model.dec_filters),
        out_channels=config.model.out_channels,
        temb_dim=config.model.temb_dim,
        healpix_emb_dim=config.model.healpix_emb_dim,
        edges_to_healpix=edges_to_healpix,
        edges_to_latlon=edges_to_latlon
    )
    print_parameter_count(model)
    # Initialize denoiser with preconditioning
    denoiser = Denoiser(model, config.model.context_channels)
    
    # Train the model
    denoiser = eqx.tree_deserialise_leaves("weights.eqx", denoiser) ## for diffusion, started with 0 weights
    denoiser = train(denoiser, train_dataset, val_dataset, schedule, μ_train, σ_train, config)
    
    # Save the trained model
    import os


    EXPERIMENT_DIR = os.path.dirname(__file__)
    CACHE_DIR = os.path.join(EXPERIMENT_DIR, "cache")
    EXPERIMENT_NAME = os.path.basename(EXPERIMENT_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)
    eqx.tree_serialise_leaves(os.path.join(CACHE_DIR, "weights_consistency.eqx"), denoiser) ## for diffusion, config.training.model_filename
    print(f"Model saved to weights_consistency.eqx")    ## for diffusion, config.training.model_filename


if __name__ == "__main__":
    main()