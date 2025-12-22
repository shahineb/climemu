# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# %%
root = "/Users/shahine/Documents/Research/MIT/code/repos/climemu-private/paper/intermodel/wandb/"
train_losses = {"MIROC6": pd.read_csv(os.path.join(root, 'miroc_train.csv')),
                "MPI-ESM1-2-LR": pd.read_csv(os.path.join(root, 'mpi_train.csv')),
                "ACCESS-ESM1-5": pd.read_csv(os.path.join(root, 'access_train.csv'))}

val_losses = {"MIROC6": pd.read_csv(os.path.join(root, 'miroc_val.csv')),
              "MPI-ESM1-2-LR": pd.read_csv(os.path.join(root, 'mpi_val.csv')),
              "ACCESS-ESM1-5": pd.read_csv(os.path.join(root, 'access_val.csv'))}

grad_df = {"MIROC6": pd.read_csv(os.path.join(root, 'miroc_grad.csv')),
           "MPI-ESM1-2-LR": pd.read_csv(os.path.join(root, 'mpi_grad.csv')),
           "ACCESS-ESM1-5": pd.read_csv(os.path.join(root, 'access_grad.csv'))}



# %%
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

train_df = train_losses['MIROC6']
val_df = val_losses['MIROC6']
ax[0].plot(train_df['Step'], train_df.iloc[:, 1], label='MIROC6 Training Loss', alpha=0.5, color='#0072B2', zorder=0)
ax[0].plot(val_df['Step'], val_df.iloc[:, 1], ls='--', label='MIROC6 Validation Loss', color='#0072B2')

train_df = train_losses['MPI-ESM1-2-LR']
val_df = val_losses['MPI-ESM1-2-LR']
ax[0].plot(train_df['Step'], train_df.iloc[:, 1], label='MPI-ESM1-2-LR Training Loss', alpha=0.5, color='#E69F00', zorder=0)
ax[0].plot(val_df['Step'], val_df.iloc[:, 1], ls='--', label='MPI-ESM1-2-LR Validation Loss', color='#E69F00')

train_df = train_losses['ACCESS-ESM1-5']
val_df = val_losses['ACCESS-ESM1-5']
ax[0].plot(train_df['Step'], train_df.iloc[:, 1], label='ACCESS-ESM1-5 Training Loss', alpha=0.5, color='#CC79A7', zorder=0)
ax[0].plot(val_df['Step'], val_df.iloc[:, 1], ls='--', label='ACCESS-ESM1-5 Validation Loss', color='#CC79A7')

ax[0].legend(frameon=False, fontsize=11, loc='lower left')
ax[0].set_yscale('log')
ax[0].set_xscale('log')
ax[0].set_xlabel("Training Steps", fontsize=14)
ax[0].set_ylabel("Loss", fontsize=14)
ax[0].margins(0.01)

grad_df_miroc = grad_df['MIROC6']
ax[1].plot(grad_df_miroc['Step'], grad_df_miroc.iloc[:, 1], label='MIROC6 Gradient Norm', alpha=0.5, color='#0072B2')

grad_df_mpi = grad_df['MPI-ESM1-2-LR']
ax[1].plot(grad_df_mpi['Step'], grad_df_mpi.iloc[:, 1], label='MPI-ESM1-2-LR Gradient Norm', alpha=0.5, color='#E69F00')

grad_df_access = grad_df['ACCESS-ESM1-5']
ax[1].plot(grad_df_access['Step'], grad_df_access.iloc[:, 1], label='ACCESS-ESM1-5 Gradient Norm', alpha=0.5, color='#CC79A7')

ax[1].legend(frameon=False, fontsize=11)
ax[1].set_yscale('log')
ax[1].set_xscale('log')
ax[1].set_xlabel("Training Steps", fontsize=14)
ax[1].set_ylabel("Gradient Norm", fontsize=14)
ax[1].margins(0.01)

plt.savefig("losses.eps", format="eps", bbox_inches="tight")

# %%
