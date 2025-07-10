""" This script calculates the average significance of several observations NGC 1068 using the CTAO """

# ------------------------------
# IMPORTS
# ------------------------------
import os
import glob
import itertools
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
# CONFIG
# ------------------------------
Nsim = 100
BASE_PATH = Path("/Users/tharacaba/Desktop/Tesis_2/MASS_Thesis/simulations/Fermi")
EVENTS_DIR = "/Users/tharacaba/Desktop/Tesis_2/MASS_Thesis/simulations/Fermi/5sims/events/"
IRF_PATH = "/Users/tharacaba/Desktop/Tesis_2/gammapy-datasets/1.3/cta-prod5-zenodo-fitsonly-v0/fits/CTA-Performance-prod5-v0.1-North-40deg.FITS/Prod5-North-40deg-AverageAz-4LSTs09MSTs.180000s-v0.1.fits"
LIVETIME = "50 hr"
Model_type = "Fermi Catalogue Best Fit PL"

# ------------------------------
# HELPERS
# ------------------------------
def save_figure(filename):       #routine for saving figures
    path = BASE_PATH / filename      #saving paths
    path.parent.mkdir(parents=True, exist_ok=True)  #create a directory
    plt.savefig(path, dpi=300, bbox_inches='tight')     #dpi is the resolution
    plt.clf()                #clear
    plt.close()

# ------------------------------
# LOAD FILES AND OBSERVATIONS
# ------------------------------
input_files = glob.glob(os.path.join(EVENTS_DIR, '*.fits'))
datastore = DataStore.from_events_files(sorted(input_files), irfs_paths=IRF_PATH)
datastore.hdu_table.write(BASE_PATH / "hdu-index.fits.gz", overwrite=True)
datastore.obs_table.write(BASE_PATH / "obs-index.fits.gz", overwrite=True)
observations = datastore.get_observations(obs_id=list(datastore.obs_table['OBS_ID']))

# ------------------------------
# GEOMETRY SETUP
# ------------------------------
TARGET_POSITION = SkyCoord(40.669, -0.013, unit="deg", frame="icrs")
ON_REGION_RADIUS = 0.11 * u.deg
on_region = CircleSkyRegion(center=TARGET_POSITION, radius=Angle(ON_REGION_RADIUS))

MAP_WIDTH = 2.0 * u.deg
MAP_BIN_SIZE = 0.02 * u.deg
ENERGY_BOUNDS = [10 * u.GeV, 10 * u.TeV]
N_ENERGY_BINS = 10
N_TRUE_ENERGY_BINS = 13
ALPHA = 0.083

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
# DATASET MAKERS
# ------------------------------
dataset_maker = SpectrumDatasetMaker(containment_correction=True, selection=["counts", "exposure", "edisp"])
background_maker = ReflectedRegionsBackgroundMaker(exclusion_mask=exclusion_mask)
safe_mask_maker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10, position=TARGET_POSITION)

# ------------------------------
# CREATE DATASETS
# ------------------------------
datasets = Datasets()
counts_map = Map.create(skydir=TARGET_POSITION, width=3 * u.deg)

for i, obs in enumerate(observations, 1):
    dataset = dataset_maker.run(dataset_empty.copy(name=str(obs.obs_id)), obs)
    counts_map.fill_events(obs.events)
    dataset = background_maker.run(dataset, obs)
    dataset = safe_mask_maker.run(dataset, obs)
    datasets.append(dataset)

    if i % 25 == 0 or i == len(observations):
        print(f"Processed {i} out of {len(observations)} observations")

# ------------------------------
# SIGNIFICANCE ANALYSIS
# ------------------------------
sig_distrib = []

for dataset in datasets:
    mask = dataset.mask_safe.quantity
    on = dataset.counts.data[mask].sum()
    off = dataset.counts_off.data[mask].sum()
    stat = WStatCountsStatistic(n_on=on, n_off=off, alpha=ALPHA)
    sig_distrib.append(stat.sqrt_ts)

sig_distrib = np.array(sig_distrib)
mean, std = sig_distrib.mean(), sig_distrib.std()

# ------------------------------
# PLOTTING
# ------------------------------
plt.figure()
plt.hist(sig_distrib, bins=10, edgecolor="black")
plt.axvline(mean, color="red", label="Mean")
plt.axvline(mean + 2 * std, color="red", linestyle="--", label="±2σ")
plt.axvline(mean - 2 * std, color="red", linestyle="--")

plt.xlabel(r"Significance (in $\sigma$)")
plt.ylabel("Counts")
plt.title(f"NGC 1068 Significance Distribution ({Model_type}, {LIVETIME}, {Nsim} sims)")
plt.legend()
plt.grid(True)
plt.tight_layout()

safe_model = Model_type.replace(" ", "_")
safe_livetime = LIVETIME.replace(" ", "_")
filename = f"{safe_model}_{safe_livetime}_{Nsim}_significance.png"
save_figure(filename)

print(f"Significance : {mean:.2f} +/- {std:.2f}")
