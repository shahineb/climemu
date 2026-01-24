"""
FaIR calibration parameters and volcanic forcing were taken from: https://github.com/OMS-NetZero/FAIR
"""
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

# Constants
base_dir = os.path.dirname(__file__)
EBM_CONFIG = os.path.join(base_dir, 'data/4xCO2_cummins_ebm3.csv')
DEFAULT_SCENARIO = 'ssp245'
VOLCANIC_FORCING = os.path.join(base_dir, 'data/volcanic_ERF_monthly_175001-201912.csv')
EMISSIONS_FILE = os.path.join(base_dir, 'data/extensions_1750-2500.csv')
OUTPUT_DIR = "/orcd/data/raffaele/001/shahineb/emulated/climemu-private/mpi/cmip7_medium"


def get_ebm_configs(esms):
    """Get energy balance model configurations for given ESMs."""
    ebm_df = pd.read_csv(EBM_CONFIG)
    ebm_configs = []
    for model in esms:
        for run in ebm_df.loc[ebm_df['model'] == model, 'run']:
            ebm_configs.append(f"{model}_{run}")
    return ebm_df, ebm_configs


def load_cmip7_co2_emissions():
    """Load CO2 emissions for CMIP7 medium scenario."""
    df = pd.read_csv(EMISSIONS_FILE)
    df = df.loc[(df.scenario == "medium-extension") & (df.variable == "CO2 FFI")]
    years = pd.to_numeric(df.columns, errors="coerce")
    year_mask = (years <= 2100.5)
    return df.loc[:, year_mask].values


def load_volcanic_forcing():
    """Load and process volcanic forcing data."""
    df_volcanic = pd.read_csv(VOLCANIC_FORCING, index_col='year')
    volcanic_forcing = np.zeros(352)
    volcanic_data = df_volcanic.loc[1749:].groupby(
        np.ceil(df_volcanic.loc[1749:].index) // 1
    ).mean().squeeze().values
    volcanic_forcing[:271] = volcanic_data
    return volcanic_forcing


def configure_ebm(f, ebm_df, config):
    """Configure energy balance model parameters."""
    model, run = config.split('_')
    condition = (ebm_df['model'] == model) & (ebm_df['run'] == run)
    fill(f.climate_configs['ocean_heat_capacity'], 
         ebm_df.loc[condition, 'C1':'C3'].values.squeeze(), config=config)
    fill(f.climate_configs['ocean_heat_transfer'], 
         ebm_df.loc[condition, 'kappa1':'kappa3'].values.squeeze(), config=config)
    fill(f.climate_configs['deep_ocean_efficacy'], 
         ebm_df.loc[condition, 'epsilon'].values[0], config=config)
    fill(f.climate_configs['gamma_autocorrelation'], 
         ebm_df.loc[condition, 'gamma'].values[0], config=config)
    fill(f.climate_configs['sigma_eta'], 
         ebm_df.loc[condition, 'sigma_eta'].values[0], config=config)
    fill(f.climate_configs['sigma_xi'], 
         ebm_df.loc[condition, 'sigma_xi'].values[0], config=config)
    fill(f.climate_configs['stochastic_run'], False, config=config)
    fill(f.climate_configs['use_seed'], False, config=config)
    fill(f.climate_configs['seed'], 42, config=config)


def run_fair():
    """Run FAIR model and return GMST anomaly."""
    # Initialize FAIR model
    f = FAIR(ghg_method="meinshausen2020", ch4_method='thornhill2021')
    f.define_time(1750, 2101, 1)

    # Load energy balance model configs
    esms = ["MPI-ESM1-2-LR"]
    ebm_df, configs = get_ebm_configs(esms)
    f.define_configs(configs)

    # Define species
    species = ['CO2', 'CH4', 'N2O', 'Volcanic']
    properties = {s: read_properties()[1][s] for s in species}
    properties['CO2']['input_mode'] = 'emissions'
    f.define_species(species, properties)

    # Load base scenario emissions (used for non-CO2 species)
    f.define_scenarios([DEFAULT_SCENARIO])
    f.allocate()
    f.fill_from_rcmip()

    # Substitute CO2 emissions with emissions from M scenario
    cmip7_co2 = load_cmip7_co2_emissions()
    f.emissions.sel(specie='CO2')[:] = cmip7_co2.reshape(-1, 1, 1)

    # Load volcanic forcing
    volcanic_forcing = load_volcanic_forcing()
    fill(f.forcing, volcanic_forcing[:, None, None], specie="Volcanic")

    # Configure species and initialize
    f.fill_species_configs()
    initialise(f.concentration, f.species_configs['baseline_concentration'])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)

    # Configure energy balance model
    configure_ebm(f, ebm_df, configs[0])

    # Run model
    f.run()
    gmst = f.temperature.isel(layer=0, config=0, scenario=0, drop=True)
    gmst = gmst.rename(timebounds="year")
    return f, gmst


def emulate_year(emulator, gmst, year, ncall, output_dir, pbar=None):
    """Emulate climate data for a given year."""
    file_path = os.path.join(output_dir, f"{year}.nc")
    if os.path.exists(file_path):
        return

    month_samples = []
    for month in range(1, 13):
        samples_calls = []
        for _ in range(ncall):
            if pbar:
                pbar.set_description(f"{month}/{year}")
            sample = emulator(gmst, month, xarray=True)
            samples_calls.append(sample)
            if pbar:
                _ = pbar.update(1)
        samples_calls = xr.concat(samples_calls, dim="member")
        month_samples.append(samples_calls)

    result = xr.concat(month_samples, dim=xr.DataArray(range(1, 13), dims="month", name="month"))
    result.to_netcdf(file_path)
    del result
    gc.collect()


def main():
    """Main execution function."""
    # Run FAIR model
    _, gmst = run_fair()
    
    # Setup emulator
    n_samples = 50
    ncall = 2
    years = [2100]
    
    emulator = climemu.build_emulator("MPI-ESM1-2-LR")
    emulator.load()
    emulator.compile(n_samples=n_samples)
    
    # Emulate years
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    n_iter = len(years) * 12 * ncall
    with tqdm(total=n_iter) as pbar:
        for year in years:
            file_path = os.path.join(OUTPUT_DIR, f"{year}.nc")
            if os.path.exists(file_path):
                pbar.update(12 * ncall)
            else:
                ΔT = gmst.sel(year=year).values.squeeze()
                emulate_year(emulator, ΔT, year, ncall, OUTPUT_DIR, pbar)


if __name__ == "__main__":
    main()