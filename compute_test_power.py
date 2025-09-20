import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
from tqdm import tqdm
import jax.random as jr
from torch.utils.data import DataLoader, Subset

base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from src.utils.collate import numpy_collate
from experiments.mpi.config import Config
from experiments.mpi.data import load_dataset, compute_normalization
import experiments.mpi.utils as utils


# Load data
config = Config()
dataset = load_dataset(
    root=config.data.root_dir,
    model=config.data.model_name,
    experiments=config.data.train_experiments,
    variables=config.data.variables,
    in_memory=config.data.in_memory,
    pattern_scaling_path=config.data.pattern_scaling_path)

μ_train, σ_train = compute_normalization(
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


# Function to estimate power of Kstest test when adding N(0, σmax²)
def estimate_power(dataset, σmax, α, n_iter, key):
    # Initialize dataloader on subset of size n_iter
    dataset_size = len(dataset)
    indices = jr.permutation(key, dataset_size)[:n_iter].tolist()
    rejections = 0
    dataset_subset = Subset(dataset, indices)
    dummy_loader = DataLoader(dataset_subset, batch_size=1, shuffle=True, collate_fn=numpy_collate)

    # Estimate power on this subset
    rejections = 0
    with tqdm(total=n_iter) as pbar:
        pbar.set_description(f"Estimating power for σmax = {σmax:.1f}")
        for batch in dummy_loader:
            # Draw sample and flatten
            x = utils.process_batch(batch, μ_train, σ_train)[:, :-ctx_size]
            x0 = np.array(x.ravel())

            # Add noise and perform test
            xn = x0 + σmax * np.random.randn(len(x0))
            _, pvalue = kstest(xn, "norm", args=(0, σmax))
            rejections += (pvalue < α)
            _ = pbar.update(1)
    return rejections / n_iter



σmax_low = 100.
σmax_high = 300.
power = 1.
key = jr.PRNGKey(2)
n_split = 10
n_iter = 100

with tqdm(total=n_split) as pbar:
    for i in range(n_split):
        # Set σmax in the middle of search interval
        σmax = 0.5 * (σmax_low + σmax_high)

        # Estimate CI on test power
        key, χ = jr.split(key)
        power = estimate_power(dataset, σmax, α=0.05, n_iter=n_iter, key=χ)
        spread = 1.96 * np.sqrt(power * (1 - power) / n_iter)
        lb, ub = power - spread, power + spread
        pbar.set_description(f"σmax = {σmax} -> Power = {power:.3f}±{spread:.3f}")

        # Update search interval
        if lb > 0.1:
            σmax_low = σmax
        elif ub < 0.1:
            σmax_high = σmax
        else:
            if (lb >= 0.09) and (ub <= 1.01):
                break
            else:
                print("Uncertain, increasing n_iter \n")
                n_iter = min(2 * n_iter, 5000)
        _ = pbar.update(1)
        if np.allclose(σmax_low, σmax_high, atol=1):
            break





# sigmas = [1, 5, 10, 20, 50, 100, 150]
# alphas = [0.1, 0.05, 0.01]
# powers = {α : [] for α in alphas}

# with tqdm(total=len(sigmas)) as pbar:
#     for α in alphas:
#         for σmax in sigmas:
#             pbar.set_description(f"σmax={σmax}")
#             power = estimate_power(dummy_loader, σmax, α)
#             powers[α].append(power)
#             print(f"(σmax={σmax}, α={α}) => Power = {power:.3f}")
#             _ = pbar.update(1)


# # Plotting
# fig, ax = plt.subplots(1, 1, figsize=(5, 3))
# for alpha, results in powers.items():
#     ax.plot(sigmas, results, marker="o", label=f"α={alpha}")
# ax.set_xlabel("σmax", weight="bold")
# ax.set_ylabel("Test power", weight="bold")
# ax.set_xscale('log')
# ax.legend(frameon=False)
# plt.savefig("test_power.jpg", dpi=300, bbox_inches="tight")
# plt.close()



# # Estimate population standard deviation for reference
# # population = []
# # for batch in tqdm(dummy_loader):
# #     x = utils.process_batch(batch, μ, σ)[:, :-ctx_size]
# #     population.append(x.ravel())
# # population = np.concatenate(population).ravel()
# # popmean = population.mean()
# # popstddev = population.std()
# # σcontrol = np.abs(popmean) + popstddev
# # print(f"Noise level we control for: {σcontrol:.4f}")



# # Function to estimate power of Kstest test when adding N(0, σ)
# def estimate_power(dataloader, n, alpha, n_iter=1000):
#     rejections = 0
#     dataset_iter = iter(dataloader)
#     for _ in tqdm(range(n_iter)):
#         # Draw sample
#         batch = next(dataset_iter)
#         x = utils.process_batch(batch, μ, σ)[:, :-ctx_size]

#         # Flatten, shuffle, and take first n samples
#         x0 = np.array(x.ravel())
#         np.random.shuffle(x0)
#         x0 = x0[:n]

#         # Add noise and perform test
#         xn = x0 + σcontrol * np.random.randn(n)
#         _, pvalue = kstest(xn, "norm", args=(0, σcontrol))
#         rejections += (pvalue < alpha)
#     return rejections / n_iter


# def estimate_typeI(n, alpha, n_iter=1000):
#     rejections = 0
#     for _ in tqdm(range(n_iter)):
#         # Draw noise and perform Anderson test
#         xn = σcontrol * np.random.randn(n)
#         _, pvalue = kstest(xn, "norm", args=(0, σcontrol))
#         rejections += (pvalue < alpha)
#     return rejections / n_iter



# # Settings
# ns = [100, 500, 1000, 5000, 10000, 20000, 73728]
# alphas = [0.1, 0.05, 0.01]

# powers = {α : [] for α in alphas}
# typeIs = {α : [] for α in alphas}

# np.random.seed(0)
# with tqdm(total=len(ns)) as pbar:
#     for alpha in alphas:
#         for n in ns:
#             pbar.set_description(f"n={n}, α={alpha}")
#             power = estimate_power(dummy_loader, n, alpha)
#             powers[alpha].append(power)
#             typeI = estimate_typeI(n, alpha)
#             typeIs[alpha].append(typeI)
#             print(f"(n={n}, α={alpha}) => Power = {power:.3f}, Type I = {typeI:.3f}")
#             _ = pbar.update(1)



# # Plotting
# fig, ax = plt.subplots(1, 2, figsize=(12, 4))
# for alpha, results in powers.items():
#     ax[0].plot(ns, results, marker="o", label=f"α={alpha}")
# ax[0].axhline(y=0.8, color="gray", linestyle="--", label="80% power (target)")
# ax[0].set_xlabel("Sample size (n)")
# ax[0].set_ylabel("P(reject|not Gaussian)")
# ax[0].set_ylim(0, 0.2)
# ax[0].set_xscale('log')
# for alpha, results in typeIs.items():
#     ax[1].plot(ns, results, marker="o", label=f"α={alpha}")
# ax[1].set_xlabel("Sample size (n)")
# ax[1].set_ylabel("P(reject|Gaussian)")
# ax[1].legend(frameon=False)
# ax[1].set_ylim(0, 0.2)
# ax[1].set_xscale('log')
# plt.savefig("test_power.jpg", dpi=300)
# plt.close()