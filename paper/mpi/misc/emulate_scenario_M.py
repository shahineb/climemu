# %%
import os
import gc
import numpy as np
import xarray as xr
import pandas as pd
from fair import FAIR
from fair.io import read_properties
from fair.interface import fill, initialise
import climemu
from tqdm import tqdm


EBM_CONFIG = './files/4xCO2_cummins_ebm3.csv'
DEFAULT_SCENARIO = 'ssp245'
VOLCANIC_FORCING = 'files/volcanic_ERF_monthly_175001-201912.csv'


def get_ebm_configs(esms):
    ebm_df = pd.read_csv(EBM_CONFIG)
    ebm_configs = []
    for model in esms:
        for run in ebm_df.loc[ebm_df['model']==model, 'run']:
            ebm_configs.append(f"{model}_{run}")
    return ebm_df, ebm_configs


# %%
def run_fair():
    # Instantiate FaIR model
    f = FAIR(ghg_method="meinshausen2020", ch4_method='thornhill2021')
    f.define_time(1750, 2101, 1)

    # Load energy balance model configs tuned for MPI
    esms = ["MPI-ESM1-2-LR"]
    ebm_df, configs = get_ebm_configs(esms)
    f.define_configs(configs)

    # Define species we work with (meinshausen2020 method requires CH4 and N2O)
    species = ['CO2', 'CH4', 'N2O', 'Volcanic']
    properties = {s: read_properties()[1][s] for s in species}
    properties['CO2']['input_mode'] = 'emissions'
    f.define_species(species, properties)

    # We use emissions from another scenario for species other than CO2
    f.define_scenarios([DEFAULT_SCENARIO])
    f.allocate()
    f.fill_from_rcmip()

    # Substitute CO2 emissions with M scenario
    df = pd.read_csv("files/extensions_1750-2500.csv")
    df = df.loc[df.scenario == "medium-extension"]
    df = df.loc[df.variable == "CO2 FFI"]
    years = pd.to_numeric(df.columns, errors="coerce")
    year_mask = (years <= 2100.5)
    cmip7_co2 = df.loc[:, year_mask].values
    f.emissions.sel(specie='CO2')[:] = cmip7_co2.reshape(-1, 1, 1)

    # Load prescribed volcanic forcing for 1750-2000
    df_volcanic = pd.read_csv(VOLCANIC_FORCING, index_col='year')
    volcanic_forcing = np.zeros(352)
    volcanic_forcing[:271] = df_volcanic.loc[1749:].groupby(np.ceil(df_volcanic.loc[1749:].index) // 1).mean().squeeze().values
    fill(f.forcing, volcanic_forcing[:, None, None], specie="Volcanic")

    # Load species config for gas cycle and radiative forcing models
    f.fill_species_configs()

    # Initialise variable at starting point
    initialise(f.concentration, f.species_configs['baseline_concentration'])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)

    # Load MPI-tuned energy balance model configs
    seed = 1355763
    config = configs[0]
    model, run = config.split('_')
    condition = (ebm_df['model']==model) & (ebm_df['run']==run)
    fill(f.climate_configs['ocean_heat_capacity'], ebm_df.loc[condition, 'C1':'C3'].values.squeeze(), config=config)
    fill(f.climate_configs['ocean_heat_transfer'], ebm_df.loc[condition, 'kappa1':'kappa3'].values.squeeze(), config=config)
    fill(f.climate_configs['deep_ocean_efficacy'], ebm_df.loc[condition, 'epsilon'].values[0], config=config)
    fill(f.climate_configs['gamma_autocorrelation'], ebm_df.loc[condition, 'gamma'].values[0], config=config)
    fill(f.climate_configs['sigma_eta'], ebm_df.loc[condition, 'sigma_eta'].values[0], config=config)
    fill(f.climate_configs['sigma_xi'], ebm_df.loc[condition, 'sigma_xi'].values[0], config=config)
    fill(f.climate_configs['stochastic_run'], False, config=config)
    fill(f.climate_configs['use_seed'], False, config=config)
    fill(f.climate_configs['seed'], seed, config=config)
    seed = seed + 399

    # Run and return GMST anomaly
    f.run()
    gmst = f.temperature.isel(layer=0, config=0, scenario=0, drop=True)
    gmst = gmst.rename(timebounds="year")
    return f, gmst


# %%
_, gmst = run_fair()

# %%
n_samples = 50
ncall = 2
emulator = climemu.build_emulator("MPI-ESM1-2-LR")
emulator.load()
emulator.compile(n_samples=n_samples)
years = [2100]


# %%
n_iter = len(years) * 12 * ncall
output_dir = "/home/shahineb/data/emulated/climemu-private/cmip7_medium-extension"
os.makedirs(output_dir, exist_ok=True)
with tqdm(total=n_iter) as pbar:
    for year in years:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{year}.nc")
        if os.path.exists(file_path):
            _ = pbar.update(12 * ncall)
        else:
            ΔT = gmst.isel(year=year)
            month_samples = []
            for month in range(1, 13):
                samples_calls = []
                for ω in range(ncall):
                    pbar.set_description(f"{month}/{year} - {ω + 1}")
                    sample = emulator(gmst, month, xarray=True)
                    samples_calls.append(sample)
                    _ = pbar.update(1)
                samples_calls = xr.concat(samples_calls, dim="member")
                month_samples.append(samples_calls)
            member_ω = xr.concat(month_samples, dim=xr.DataArray(range(1, 13), dims="month", name="month"))
            member_ω.to_netcdf(file_path)
            del member_ω
            gc.collect()