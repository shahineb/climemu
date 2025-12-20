import numpy as np
import jax.numpy as jnp
import jax.random as jr
from tqdm import tqdm
from scipy.stats import kstest
from torch.utils.data import DataLoader, Subset


from paper.mpi.config import Config
from paper.mpi.data import load_dataset, compute_normalization
from paper.mpi import utils
from src.utils.collate import numpy_collate



config = Config()
train_dataset = load_dataset(
        root=config.data.root_dir,
        model=config.data.model_name,
        experiments=["piControl", "historical", "ssp585"],
        variables=config.data.variables,
        in_memory=config.data.in_memory,
        pattern_scaling_path=config.data.pattern_scaling_path
    )

μ_train, σ_train = compute_normalization(
        train_dataset,
        config.training.batch_size,
        max_samples=config.data.norm_max_samples,
        seed=config.training.random_seed,
        norm_stats_path=config.data.norm_stats_path
    )



# Load a random subset of the training data to estimate PC1
ctx_size = 1
dataset_size = len(train_dataset)
subset_size = min(10000, dataset_size)
key = jr.PRNGKey(42)
indices = jr.permutation(key, dataset_size)[:subset_size].tolist()
dataset_subset = Subset(train_dataset, indices)
dummy_loader = DataLoader(
    dataset_subset,
    batch_size=10,
    shuffle=False,
    collate_fn=numpy_collate
)
X = []
for batch in tqdm(dummy_loader, desc=f"Loading {subset_size} samples"):
    X.append(utils.process_batch(batch, μ_train, σ_train)[:, :-ctx_size])
X = jnp.concatenate(X)


# Estimate principal component 1
μX = X.mean(axis=0)
Xc = X - μX
lat = train_dataset.cmip6data.lat
wlat = jnp.cos(jnp.deg2rad(lat))
G = jnp.einsum("nchw,h,mchw->nm", Xc, wlat, Xc)
Σ2, U = jnp.linalg.eigh(G)
u1 = U[:, -1]
σ1 = jnp.sqrt(Σ2[-1])
v1 = jnp.einsum("nchw,n->chw", Xc, u1) / σ1
v1 = v1 * wlat[:, None]
v1 = v1.ravel()


def estimate_power(dataset, σmax, α, n_montecarlo, popsize, μ, σ, ctx_size, key):
    # Initialize dataloader on subset of size n_iter
    dataset_size = len(dataset)
    indices = jr.permutation(key, dataset_size)[:n_montecarlo * popsize].tolist()
    rejections = 0
    dataset_subset = Subset(dataset, indices)
    dummy_loader = DataLoader(dataset_subset, batch_size=popsize, shuffle=True, collate_fn=numpy_collate)

    # Estimate power on this subset
    rejections = 0
    with tqdm(total=n_montecarlo) as pbar:
        pbar.set_description(f"Estimating power for σmax = {σmax:.1f}")
        for batch in dummy_loader:
            # Draw sample
            x = utils.process_batch(batch, μ, σ)[:, :-ctx_size]
            x0 = np.array(x - μX).reshape(popsize, -1)

            # Add noise and project against lead PC
            xn = x0 + σmax * np.random.randn(*x0.shape)
            xnTv1 = xn @ v1

            # Perform test
            _, pvalue = kstest(xnTv1, "norm", args=(0, σmax))
            rejections += (pvalue < α)
            _ = pbar.update(1)
    return rejections / n_montecarlo



# Define search parameters
σmax_low, σmax_high = (0, 800)
max_split = 20
n_montecarlo = 100
max_montecarlo = 10000
popsize = 128
tgt_pow = 0.1
tol = 0.001 + 1.96 * np.sqrt(tgt_pow * (1 - tgt_pow)  / max_montecarlo)
key = jr.PRNGKey(42)
ctx_size = 1
alpha = 0.05


# Select σmax such that test power < 0.1
with tqdm(total=max_split) as pbar:
    for _ in range(max_split):
        # Set σmax in the middle of search interval
        σmax = 0.5 * (σmax_low + σmax_high)

        # Estimate CI on test power
        key, χ = jr.split(key)
        power = estimate_power(dataset=train_dataset,
                                     σmax=σmax,
                                     α=alpha,
                                     n_montecarlo=n_montecarlo,
                                     popsize=popsize,
                                     μ=μ_train,
                                     σ=σ_train,
                                     ctx_size=ctx_size,
                                     key=χ)
        spread = 1.96 * np.sqrt(power * (1 - power) / n_montecarlo)
        lb, ub = power - spread, power + spread
        pbar.set_description(f"σmax = {σmax} -> Power ∈ ({lb:.3f}, {ub:.3f})")
        _ = pbar.update(1)

        # Case 1: CI is fully below 0.1
        if ub < tgt_pow:
            # Early stop if CI is tight and close to 0.1 from below
            if (spread < tol) and (ub + tol / 2 > tgt_pow):
                break
            # Else look for smaller values
            σmax_high = σmax
        # Case 2: CI is fully above 0.1 OR CI is tight and straddles 0.1
        elif (lb > tgt_pow) or ((spread < tol) and (lb < tgt_pow) and (ub > tgt_pow)):
            # Look for larger values
            σmax_low = σmax
        # Case 3: Ambiguous overlap with 0.1 and CI not tight enough
        else:
            print("Uncertain, increasing nb of monte carlo samples \n")
            n_montecarlo = min(2 * n_montecarlo, max_montecarlo)
        if np.allclose(σmax_low, σmax_high, atol=1):
            break