"""
Calculate the average significance of multiple CTAO simulations of NGC 1068.
"""

# ------------------------------
# IMPORTS
# ------------------------------
import os
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.table import Table

from regions import CircleSkyRegion

from gammapy.data import DataStore
from gammapy.maps import MapAxis, WcsGeom, Map, RegionGeom
from gammapy.datasets import Datasets, SpectrumDataset
from gammapy.makers import (
    SpectrumDatasetMaker,
    ReflectedRegionsBackgroundMaker,
    SafeMaskMaker
)
from gammapy.stats import WStatCountsStatistic

# ------------------------------
# CONFIGURATION
# ------------------------------
Nsim = 100
LIVETIME = "100 hr"
MODEL_TYPE = "Peretti UFO Model"
BASE_PATH = Path("/Users/tharacaba/Desktop/Tesis_2/MASS_Thesis/simulations/UFO")
EVENTS_DIR = BASE_PATH / f"{Nsim}sims/events/"
IRF_PATH = Path("/Users/tharacaba/Desktop/Tesis_2/gammapy-datasets/1.3/cta-prod5-zenodo-fitsonly-v0/fits/CTA-Performance-prod5-v0.1-North-40deg.FITS/Prod5-North-40deg-AverageAz-4LSTs09MSTs.180000s-v0.1.fits")
ALPHA = 0.083

# ------------------------------
# UTILITY FUNCTIONS
# ------------------------------
def print_info(message):
    print(f"\033[96m {message} \033[0m")

def save_figure(filename):
    """Save and clear the current Matplotlib figure."""
    path = BASE_PATH / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()

def load_observations(events_dir, irf_path):
    """Load observations from event files and attach IRFs."""
    input_files = sorted(glob.glob(str(events_dir / "*.fits")))
    datastore = DataStore.from_events_files(input_files, irfs_paths=irf_path)
    datastore.hdu_table.write(BASE_PATH / "hdu-index.fits.gz", overwrite=True)
    datastore.obs_table.write(BASE_PATH / "obs-index.fits.gz", overwrite=True)
    return datastore.get_observations(obs_id=list(datastore.obs_table["OBS_ID"]))

# ------------------------------
# LOAD OBSERVATIONS
# ------------------------------
observations = load_observations(EVENTS_DIR, IRF_PATH)
print_info(f"Loaded {len(observations)} observations.")

# ------------------------------
# GEOMETRY SETUP
# ------------------------------
TARGET_POSITION = SkyCoord(40.669, -0.013, unit="deg", frame="icrs")
ON_REGION_RADIUS = 0.11 * u.deg
on_region = CircleSkyRegion(center=TARGET_POSITION, radius=Angle(ON_REGION_RADIUS))

MAP_WIDTH = 2.0 * u.deg
MAP_BIN_SIZE = 0.02 * u.deg
N_ENERGY_BINS = 10
N_TRUE_ENERGY_BINS = 13
ENERGY_BOUNDS = [10 * u.GeV, 10 * u.TeV]

n_pix = int((MAP_WIDTH / MAP_BIN_SIZE).decompose().value)
geom_excl = WcsGeom.create(
    npix=(n_pix, n_pix),
    binsz=MAP_BIN_SIZE,
    skydir=TARGET_POSITION.galactic,
    proj="TAN",
    frame="galactic"
)
exclusion_mask = ~geom_excl.region_mask([on_region])

energy_axis = MapAxis.from_energy_bounds(*ENERGY_BOUNDS, nbin=N_ENERGY_BINS, per_decade=True, unit="TeV", name="energy")
energy_axis_true = MapAxis.from_energy_bounds(
    ENERGY_BOUNDS[0] * 0.8,
    ENERGY_BOUNDS[1] * 1.2,
    nbin=N_TRUE_ENERGY_BINS,
    per_decade=True,
    unit="TeV",
    name="energy_true"
)

geom = RegionGeom.create(region=on_region, axes=[energy_axis])
dataset_empty = SpectrumDataset.create(geom=geom, energy_axis_true=energy_axis_true)

# ------------------------------
# DATASET CREATION
# ------------------------------
dataset_maker = SpectrumDatasetMaker(containment_correction=True, selection=["counts", "exposure", "edisp"])
background_maker = ReflectedRegionsBackgroundMaker(exclusion_mask=exclusion_mask)
safe_mask_maker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10, position=TARGET_POSITION)

datasets = Datasets()
counts_map = Map.create(skydir=TARGET_POSITION, width=3 * u.deg)

for i, obs in enumerate(observations, start=1):
    dataset = dataset_maker.run(dataset_empty.copy(name=str(obs.obs_id)), obs)
    counts_map.fill_events(obs.events)
    dataset = background_maker.run(dataset, obs)
    dataset = safe_mask_maker.run(dataset, obs)
    datasets.append(dataset)

    if i % 25 == 0 or i == len(observations):
        print_info(f"Processed {i} out of {len(observations)} observations.")

# ------------------------------
# SIGNIFICANCE ANALYSIS
# ------------------------------
sig_distrib = []

for dataset in datasets:
    mask = dataset.mask_safe.quantity
    n_on = dataset.counts.data[mask].sum()
    n_off = dataset.counts_off.data[mask].sum()
    stat = WStatCountsStatistic(n_on=n_on, n_off=n_off, alpha=ALPHA)
    sig_distrib.append(stat.sqrt_ts)

sig_distrib = np.array(sig_distrib)
mean_sig, std_sig = sig_distrib.mean(), sig_distrib.std()

# ------------------------------
# PLOTTING RESULTS
# ------------------------------
plt.figure()
plt.hist(sig_distrib, bins=10, edgecolor="black")
plt.axvline(mean_sig, color="red", label="Mean")
plt.axvline(mean_sig + 2 * std_sig, color="red", linestyle="--", label="±2σ")
plt.axvline(mean_sig - 2 * std_sig, color="red", linestyle="--")

plt.xlabel(r"Significance (in $\sigma$)")
plt.ylabel("Counts")
plt.title(f"NGC 1068 Significance Distribution ({MODEL_TYPE}, {LIVETIME}, {Nsim} sims)")
plt.legend()
plt.grid(True)
plt.tight_layout()

filename = f"{MODEL_TYPE.replace(' ', '_')}_{LIVETIME.replace(' ', '_')}_{Nsim}_significance.png"
save_figure(filename)

print_info(f"Mean Significance: {mean_sig:.2f} ± {std_sig:.2f}")