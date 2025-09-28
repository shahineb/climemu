import matplotlib.pyplot as plt
from src.datasets import CMIP6Data
from src.utils import arrays

dataset = CMIP6Data(root="/orcd/data/raffaele/001/shahineb/cmip6/processed",
                    model="MPI-ESM1-2-LR",
                    experiments=["historical", "ssp126", "ssp245", "ssp370", "ssp585"],
                    variables=["tas"])


tas = dataset.dtree.map_over_datasets(arrays.filter_var('tas'))
gmst = tas.map_over_datasets(arrays.global_mean).compute()

historical = gmst['/historical'].ds.groupby('time.year').mean()
ssp126 = gmst['/ssp126'].ds.groupby('time.year').mean()
ssp245 = gmst['/ssp245'].ds.groupby('time.year').mean()
ssp370 = gmst['/ssp370'].ds.groupby('time.year').mean()
ssp585 = gmst['/ssp585'].ds.groupby('time.year').mean()


μ = {"historical": historical.mean('member'),
     "SSP1-2.6": ssp126.mean('member'),
     "SSP2-4.5": ssp245.mean('member'),
     "SSP3-7.0": ssp370.mean('member'),
     "SSP5-8.5": ssp585.mean('member')}

σ = {"historical": historical.std('member'),
     "SSP1-2.6": ssp126.std('member'),
     "SSP2-4.5": ssp245.std('member'),
     "SSP3-7.0": ssp370.std('member'),
     "SSP5-8.5": ssp585.std('member')}

color = {"historical": "C7",
        "SSP1-2.6": "C2",
        "SSP2-4.5": "C0",
        "SSP3-7.0": "C3",
        "SSP5-8.5": "C4"}

fig, ax = plt.subplots(1, 1, figsize=(6, 3))
for i, scenario in enumerate(μ.keys()):
    year = μ[scenario].year.values
    mean = μ[scenario].tas.values
    stddev = σ[scenario].tas.values
    ub, lb = mean + 2 * stddev, mean - 2 * stddev
    ax.plot(year, mean, label=scenario, color=color[scenario])
    ax.fill_between(year, lb, ub, alpha=0.2, color=color[scenario])

ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
ax.set_ylabel("GMST anomaly [°C]", fontsize=14)
ax.legend(frameon=False, prop={"size": 14, "weight": "bold"})
ax.margins(0.001)
ax.spines['top'].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(True)
ax.tick_params(axis="both", which="major", labelsize=14) 
ax.set_xlim(1850, 2105)
ax.set_xticks([1900, 2000, 2100])
plt.savefig("ssps.jpg", dpi=300, bbox_inches='tight')
plt.close()