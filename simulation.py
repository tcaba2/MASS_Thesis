""" This script simulates observations of NGC 1068 using the CTAO """

# ========================== Imports ==========================
import os
from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.table import Table
from astropy.io import fits, ascii

from regions import CircleSkyRegion

import warnings

import gammapy
from gammapy.utils.deprecation import GammapyDeprecationWarning
from gammapy.data import Observation, observatory_locations
from gammapy.maps import MapAxis, WcsGeom, Map
from gammapy.irf import load_irf_dict_from_file
from gammapy.makers import MapDatasetMaker
from gammapy.modeling.models import (
    Models, SkyModel, FoVBackgroundModel, PointSpatialModel,
    EBLAbsorptionNormSpectralModel
)
from gammapy.datasets import MapDataset
from gammapy.catalog import CATALOG_REGISTRY
from gammapy.datasets import MapDatasetEventSampler

# Hide Gammapy warnings
warnings.filterwarnings("ignore", category=GammapyDeprecationWarning)

# ========================== CONFIG ==========================

BASE_PATH = Path("/Users/tharacaba/Desktop/Tesis_2/MASS_Thesis/simulations/Fermi")
Nsim = 10
LIVETIME = 50 * u.hr
SOURCE_NAME_AN = "NGC1068_Fermi"
IRF_FILENAME = Path("/Users/tharacaba/Desktop/Tesis_2/gammapy-datasets/1.3/cta-prod5-zenodo-fitsonly-v0/fits/CTA-Performance-prod5-v0.1-North-40deg.FITS/Prod5-North-40deg-AverageAz-4LSTs09MSTs.180000s-v0.1.fits")

# Spectral model
catalog_4fgl = CATALOG_REGISTRY.get_cls("4fgl")()
source_4fgl = catalog_4fgl["4FGL J0242.6-0000"]
fermi_model = source_4fgl.sky_model()

redshift = 0.00379
ebl = EBLAbsorptionNormSpectralModel.read_builtin("dominguez", redshift=redshift)
spectral_model = fermi_model.spectral_model * ebl

# ========================== HELPERS ==========================
def make_dirs():
    for sub in ["events", "best-fit", "spectra"]:
        dir_path = BASE_PATH / f"{Nsim}sims" / sub
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory {dir_path} is ready")

# ========================== MAIN WORKFLOW ==========================

# Setup simulation directories
make_dirs()

# Define geometry
energy_axis = MapAxis.from_energy_bounds("0.1 TeV", "100.0 TeV", nbin=10, per_decade=True)
energy_axis_true = MapAxis.from_energy_bounds("0.01 TeV", "350 TeV", nbin=20, per_decade=True, name="energy_true")
migra_axis = MapAxis.from_bounds(0.5, 2, nbin=150, node_type="edges", name="migra")

SOURCE_NAME = "NGC1068"

center = SkyCoord.from_name(SOURCE_NAME).icrs
offset = 0.5 * u.deg
pointing_position = center.directional_offset_by(position_angle=0 * u.deg, separation=offset)

geom = WcsGeom.create(
    skydir=pointing_position, width=(10, 10), binsz=0.02,
    frame="icrs", axes=[energy_axis]
)

# Load IRFs
irfs = load_irf_dict_from_file(IRF_FILENAME)

# Create empty dataset
empty_dataset = MapDataset.create(
    geom, energy_axis_true=energy_axis_true, migra_axis=migra_axis, name="my-dataset"
)

# Simulate observation for a single exposure
observation = Observation.create(
    obs_id="0001", pointing=pointing_position, livetime=LIVETIME, irfs=irfs
)

maker = MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"])
dataset = maker.run(empty_dataset, observation)
dataset.write(BASE_PATH / f"dataset_{SOURCE_NAME}_{int(LIVETIME.value)}hr.fits", overwrite=True)

spatial_model = PointSpatialModel(
    lon_0=center.ra, lat_0=center.dec, frame="icrs"
)
spatial_model.parameters["lon_0"].frozen = True
spatial_model.parameters["lat_0"].frozen = True

sky_model = SkyModel(
    spectral_model=spectral_model,
    spatial_model=spatial_model,
    name=SOURCE_NAME_AN
)

bkg_model = FoVBackgroundModel(dataset_name="my-dataset")
models = Models([sky_model, bkg_model])
models.write(BASE_PATH / f"./{SOURCE_NAME_AN}.yaml", overwrite=True)

# Load dataset again and assign models
dataset = MapDataset.read(BASE_PATH / f'dataset_{SOURCE_NAME}_{int(LIVETIME.value)}hr.fits', name="my-dataset")
dataset.models = models

# Run simulation loop
for i in range(Nsim):
    obs = Observation.create(
        obs_id=i,
        pointing=pointing_position,
        livetime=LIVETIME,
        irfs=irfs,
        tstart=30 * u.min,
        reference_time=Time("2026-05-28T00:00:00", format="isot", scale="utc"),
        location=observatory_locations["cta_north"]
    )
    
    sampler = MapDatasetEventSampler(random_state=i)
    events = sampler.run(dataset, obs)
    obs.events = events
    
    events_file = BASE_PATH / f"{Nsim}sims/events/{SOURCE_NAME}_{int(LIVETIME.value)}hr_events_{i}.fits"
    obs.write(events_file, include_irfs=False, overwrite=True)
