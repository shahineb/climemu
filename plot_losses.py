# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# %%
train_losses = {"MIROC6": pd.read_csv('miroc_train.csv'),
                "MPI-ESM1-2-LR": pd.read_csv('mpi_train.csv')}

val_losses = {"MIROC6": pd.read_csv('miroc_val.csv'),
              "MPI-ESM1-2-LR": pd.read_csv('mpi_val.csv')}

grad_df = {"MIROC6": pd.read_csv('miroc_grad.csv'),
           "MPI-ESM1-2-LR": pd.read_csv('mpi_grad.csv')}



# %%
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

train_df = train_losses['MIROC6']
val_df = val_losses['MIROC6']
ax[0].plot(train_df['Step'], train_df.iloc[:, 1], label='MIROC6 Training Loss', color='dodgerblue', zorder=0)
ax[0].scatter(val_df['Step'], val_df.iloc[:, 1], label='MIROC6 Validation Loss', color='blue')

train_df = train_losses['MPI-ESM1-2-LR']
val_df = val_losses['MPI-ESM1-2-LR']
ax[0].plot(train_df['Step'], train_df.iloc[:, 1], label='MPI-ESM1-2-LR Training Loss', color='salmon', zorder=0)
ax[0].scatter(val_df['Step'], val_df.iloc[:, 1], label='MPI-ESM1-2-LR Validation Loss', color='red')

ax[0].legend(frameon=False, fontsize=14)
ax[0].set_yscale('log')
ax[0].set_xscale('log')
ax[0].set_xlabel("Training Steps", fontsize=14)
ax[0].set_ylabel("Loss", fontsize=14)

grad_df_miroc = grad_df['MIROC6']
ax[1].plot(grad_df_miroc['Step'], grad_df_miroc.iloc[:, 1], label='MIROC6 Gradient Norm', color='dodgerblue')

grad_df_mpi = grad_df['MPI-ESM1-2-LR']
ax[1].plot(grad_df_mpi['Step'], grad_df_mpi.iloc[:, 1], label='MPI-ESM1-2-LR Gradient Norm', color='salmon')
ax[1].legend(frameon=False, fontsize=14)
ax[1].set_yscale('log')
ax[1].set_xscale('log')
ax[1].set_xlabel("Training Steps", fontsize=14)
ax[1].set_ylabel("Gradient Norm", fontsize=14)

plt.savefig("losses.eps", format="eps", bbox_inches="tight")

# %%
