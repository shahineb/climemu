# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from tqdm import tqdm
import jax.random as jr
from torch.utils.data import DataLoader, Subset
from src.utils.collate import numpy_collate
from src.experiments.mpi.config import Config
from src.experiments.mpi.data import load_dataset, compute_normalization
import src.experiments.mpi.utils as utils


# %%
# Load data
config = Config()
dataset = load_dataset(
    root=config.data.root_dir,
    model=config.data.model_name,
    experiments=config.data.train_experiments,
    variables=config.data.variables,
    in_memory=config.data.in_memory,
    pattern_scaling_path=config.data.pattern_scaling_path)

μ, σ = compute_normalization(
        dataset,
        config.training.batch_size,
        max_samples=config.data.norm_max_samples,
        seed=config.training.random_seed,
        norm_stats_path=config.data.norm_stats_path)

dataset_size = len(dataset)
subset_size = min(config.data.norm_max_samples, dataset_size)
key = jr.PRNGKey(config.training.random_seed)
indices = jr.permutation(key, dataset_size)[:subset_size].tolist()
dataset_subset = Subset(dataset, indices)
dummy_loader = DataLoader(
    dataset_subset,
    batch_size=1,
    shuffle=True,
    collate_fn=numpy_collate
)
ctx_size = config.model.context_channels

# %%
# Estimate population standard deviation for reference
population = []
for batch in dummy_loader:
    x = utils.process_batch(batch, μ, σ)[:, :-ctx_size]
    population.append(x.ravel())
population = np.concatenate(population)
σ_pop = population.std()
print(f"Estimated population stddev: {σ_pop:.4f}")


# %%
# Function to estimate power of Shapiro-Wilk test when adding N(0, σ)
def estimate_power(dataloader, n, alpha, n_iter=5000):
    rejections = 0
    dataset_iter = iter(dataloader)
    for _ in tqdm(range(n_iter)):
        # Draw sample
        batch = next(dataset_iter)
        x = utils.process_batch(batch, μ, σ)[:, :-ctx_size]

        # Flatten, shuffle, and take first n samples
        x0 = x.ravel()
        np.random.shuffle(x0)
        x0 = x0[:n]

        # Add noise and perform Shapiro-Wilk test
        xn = x0 + σ * np.random.normal(n)
        _, p = shapiro(xn)
        if p < alpha:
            rejections += 1
    return rejections / n_iter


# Settings
np.random.seed(0)
ns = [20, 50, 100, 200, 500]
alphas = [0.01, 0.05, 0.1]


# %%
powers = []
with tqdm(total=len(ns) * len(alphas)) as pbar:
    for n in ns:
        for alpha in alphas:
            pbar.set_description(f"n={n}, alpha={alpha}")
            power = estimate_power(dataloader, n, alpha)
            powers.append((n, alpha, power))
            _ = pbar.update(1)



# %%
# Run simulations
results = {name: {alpha: [] for alpha in alphas} for name in alternatives}
for alt_name, dist in alternatives.items():
    for alpha in alphas:
        for n in ns:
            power = estimate_power(dist, n, alpha)
            results[alt_name][alpha].append(power)


# %%
# Plotting
plt.figure(figsize=(10, 6))

for alt_name, dist_results in results.items():
    for alpha, powers in dist_results.items():
        plt.plot(ns, powers, marker="o", label=f"{alt_name}, α={alpha}")

plt.axhline(y=0.8, color="gray", linestyle="--", label="80% power (target)")
plt.xlabel("Sample size (n)")
plt.ylabel("Estimated power")
plt.title("Power of Shapiro-Wilk test vs sample size")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
