import copy
from collections import deque
from functools import partial
from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from tqdm import tqdm
import wandb

from src.diffusion import denoising_make_step_doy, denoising_batch_loss_doy
from src.datasets import PatternToDayCMIP6Data
from paper.mpi.config import Config
from .data import make_dataloader
from . import utils


@dataclass
class TrainingState:
    """State maintained during training.
    
    Tracks the model, optimizer state, and training progress.
    
    Attributes:
        model: The current model parameters
        opt_state: The optimizer state
        step: Global step counter
        epoch: Current epoch number
    """
    model: eqx.Module
    ema_model: eqx.Module
    opt_state: optax.OptState
    step: int = 0
    epoch: int = 0


def train_epoch(
    state: TrainingState,
    train_loader: object,
    val_loader: object,
    schedule: object,
    μ: jnp.ndarray,
    σ: jnp.ndarray,
    log_sampler: callable,
    log_target_data: jnp.ndarray,
    config: Config,
    optimizer: optax.GradientTransformation
) -> TrainingState:
    """Train for one epoch and evaluate.
    
    Performs a full training epoch followed by validation.
    
    Args:
        state: Current training state
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        schedule: Noise schedule for diffusion
        μ: Mean for data normalization
        σ: Standard deviation for data normalization
        log_sampler: Sampling function for logging
        log_target_data: Target data for visualization
        config: Training configuration
        optimizer: Optimization algorithm
        
    Returns:
        Updated training state
    """
    # Initialize sliding windows for metrics
    loss_queue = deque(maxlen=config.training.queue_length)
    grad_queue = deque(maxlen=config.training.queue_length)
    
    # Setup random keys for training and validation
    χtrain, χval = jr.split(jr.PRNGKey(state.epoch + 1), 2)
    
    # Calculate steps per epoch
    n_train_steps = len(train_loader)
    n_val_steps = len(val_loader)
    
    # Training phase
    with tqdm(total=n_train_steps) as pbar:
        for batch in train_loader:
            # Process batch and normalize
            doy, x = utils.process_batch(batch, μ, σ)
            _, χtrain = jr.split(χtrain)
            
            # Perform a single optimization step
            value, model, χtrain, opt_state, grad_norm = denoising_make_step_doy(
                state.model, config.model.context_channels, schedule, x, doy, χtrain, state.opt_state, optimizer.update
            )

            # Update training state
            ema_model = utils.update_ema(state.ema_model, model, config.training.ema_decay)
            state = TrainingState(model, ema_model, opt_state, state.step + 1, state.epoch)
            
            # Track metrics in sliding windows
            loss_queue.append(value.item())
            grad_queue.append(grad_norm.item())
            running_loss = sum(loss_queue) / len(loss_queue)
            running_grad = sum(grad_queue) / len(grad_queue)

            # Update progress bar
            pbar.set_description(f"Epoch {state.epoch + 1} | Loss {round(running_loss, 2)}")
            _ = pbar.update(1)
  
            # Log metrics at specified intervals
            if (state.step + 1) % config.training.log_interval == 0:
                wandb.log({
                    "Train Loss": running_loss, 
                    "Gradient norm": running_grad
                }, step=state.step)
   
            # Generate and log samples at specified intervals
            if (state.step + 1) % config.training.sample_interval == 0:
                # Generate samples from current model
                pred_samples = log_sampler(model=ema_model, key=χtrain)

                # Log samples and metrics to wandb
                utils.log_samples(pred_samples, log_target_data, config.data.variables, state.step)

    # Validation phase
    # val_loss = 0
    # with tqdm(total=n_val_steps, desc="Evaluation") as pbar:        
    #     for batch_idx, batch in enumerate(val_loader):
    #         # Process batch and compute validation loss
    #         doy, x = utils.process_batch(batch, μ, σ)
    #         val_value = denoising_batch_loss_doy(
    #             state.ema_model, config.model.context_channels, schedule, x, doy, χval
    #         )
    #         val_loss += val_value.item()
    #         # Update progress bar
    #         pbar.set_description(f"Epoch {state.epoch + 1} | Val {round(val_loss / (batch_idx + 1), 2)}")
    #         pbar.update(1)

    # Log validation loss
    # wandb.log({"Validation Loss": val_loss / n_val_steps}, step=state.step)

    # Checkpoint weights
    if (state.epoch + 1) % config.training.checkpoint_interval == 0:
        eqx.tree_serialise_leaves(config.training.checkpoint_filename, state.ema_model)

    # Update epoch counter and return updated state
    return TrainingState(state.model, state.ema_model, state.opt_state, state.step, state.epoch + 1)


def train(
    model: eqx.Module,
    train_dataset: PatternToDayCMIP6Data,
    val_dataset: PatternToDayCMIP6Data,
    schedule: object,
    μ: jnp.ndarray,
    σ: jnp.ndarray,
    config: Config
) -> eqx.Module:
    """Train the model for the specified number of epochs.
    
    Orchestrates the full training process including initialization,
    epoch iterations, and logging.
    
    Args:
        model: Initial model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        schedule: Noise schedule for diffusion
        μ: Mean for data normalization
        σ: Standard deviation for data normalization
        config: Training configuration
        
    Returns:
        Trained model
    """
    # Setup optimizer
    optimizer = optax.adam(learning_rate=config.training.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    
    # Initialize training state
    ema_model = copy.deepcopy(model)
    state = TrainingState(model, ema_model, opt_state)

    # Create data loaders with numpy collate function
    train_loader = make_dataloader(
        train_dataset,
        batch_size=config.training.batch_size
    )

    # val_loader = make_dataloader(
    #     val_dataset,
    #     batch_size=config.training.batch_size
    # )

    # Get a single batch used visualization and metrics logging
    log_doy, log_pattern, log_target_data = utils.get_sample_batch(
        dataset=train_dataset,
        batch_size=16,
        key=jr.PRNGKey(config.training.random_seed)
    )
    log_sampler = partial(
        utils.draw_samples_batch,
        schedule=schedule,
        doy_batch=log_doy,
        pattern_batch=log_pattern,
        n_samples=config.training.sample_count,
        n_steps=config.training.sample_steps,
        μ=μ,
        σ=σ,
        output_size=(config.model.out_channels, config.model.input_size[1], config.model.input_size[2])
    )

    # Initialize wandb for experiment tracking
    wandb.init(project=config.training.wandb_project, config=config)

    # Log initial context for reference
    utils.log_initial_context(log_pattern)

    # Training loop - iterate through epochs
    for _ in range(config.training.epochs):
        state = train_epoch(
            state, train_loader, None, schedule,
            μ, σ, log_sampler, log_target_data, config, optimizer
        )
    
    # Finish wandb run
    wandb.finish()
    
    # Return the EMA trained model
    return state.ema_model