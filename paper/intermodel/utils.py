"""
Utility functions for intermodel plotting scripts.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def setup_figure(width_ratios, height_ratios, width_multiplier=1.0, height_multiplier=1.0, wspace=0.1, hspace=0.1):
    """
    Setup figure with specified grid layout and sizing.
    
    Args:
        width_ratios: List of width ratios for columns
        height_ratios: List of height ratios for rows
        width_multiplier: Multiplier for figure width
        height_multiplier: Multiplier for figure height
        wspace: Width spacing between subplots
        hspace: Height spacing between subplots
    
    Returns:
        fig: matplotlib Figure object
        gs: GridSpec object
    """
    nrow = len(height_ratios)
    ncol = len(width_ratios)
    nroweff = sum(height_ratios)
    ncoleff = sum(width_ratios)
    
    fig = plt.figure(figsize=(width_multiplier * ncoleff, height_multiplier * nroweff))
    gs = gridspec.GridSpec(nrows=nrow,
                          ncols=ncol,
                          figure=fig,
                          width_ratios=width_ratios,
                          height_ratios=height_ratios,
                          hspace=hspace,
                          wspace=wspace)
    
    return fig, gs


def save_plot(fig, output_dir, filename, dpi=300):
    """
    Save plot to specified directory.
    
    Args:
        fig: matplotlib Figure object
        output_dir: Directory to save the plot
        filename: Name of the file to save
        dpi: DPI for the saved plot
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
