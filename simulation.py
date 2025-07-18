"""
Simulate CTAO observations of NGC 1068 using different spectral models.
"""

# ========================== Imports ==========================
import os
from pathlib import Path
import warnings

from math import pi

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.io import ascii

from gammapy.utils.deprecation import GammapyDeprecationWarning
from gammapy.data import Observation, observatory_locations
from gammapy.maps import MapAxis, WcsGeom
from gammapy.irf import load_irf_dict_from_file
from gammapy.makers import MapDatasetMaker
from gammapy.modeling.models import (
    Models, SkyModel, FoVBackgroundModel,
    PointSpatialModel, EBLAbsorptionNormSpectralModel, TemplateSpectralModel
)
from gammapy.datasets import MapDataset, MapDatasetEventSampler
from gammapy.catalog import CATALOG_REGISTRY

warnings.filterwarnings("ignore", category=GammapyDeprecationWarning)

# ========================== Config ==========================

BASE_PATH = Path("/Users/tharacaba/Desktop/Tesis_2/MASS_Thesis/simulations/Eichmann_starburst")
Nsim = 100
LIVETIME = 50 * u.hr
SOURCE_NAME_AN = "NGC1068_Eichmann"
IRF_FILENAME = Path("/Users/tharacaba/Desktop/Tesis_2/gammapy-datasets/1.3/cta-prod5-zenodo-fitsonly-v0/fits/CTA-Performance-prod5-v0.1-North-40deg.FITS/Prod5-North-40deg-AverageAz-4LSTs09MSTs.180000s-v0.1.fits")

# ------------------ Load Spectral Model ------------------
# Template spectral model. Defined by values from Eichmann+ 2022
data = ascii.read("/Users/tharacaba/Desktop/Tesis_2/MASS_Thesis/simulations/Eichmann_starburst/Eichmann_starburst.csv")

energy = data['x'] *u.GeV
values = data['y'] *u.eV / (u.cm **2.0 * u.s)

energy_MeV = energy.to(u.MeV)

values = values.to(u.MeV / (u.cm ** 2.0 * u.s))
values_MeV = values / energy_MeV**2  # divide by energy to get dN/dE

spectral_model = TemplateSpectralModel(energy=energy_MeV, values=values_MeV)

# ========================== Helper Functions ==========================
def make_dirs(base_path: Path, nsim: int):
    """Create output directories for simulations."""
    for sub in ["events", "best-fit", "spectra"]:
        dir_path = base_path / f"{nsim}sims" / sub
        os.makedirs(dir_path, exist_ok=True)
        print(f"\033[96m Directory created: {dir_path} \033[0m")

# ========================== Main Workflow ==========================

# Create output directories
make_dirs(BASE_PATH, Nsim)

# ------------------ Define Map Geometry ------------------
energy_axis = MapAxis.from_energy_bounds("0.1 TeV", "100 TeV", nbin=10, per_decade=True)
energy_axis_true = MapAxis.from_energy_bounds("0.01 TeV", "350 TeV", nbin=20, per_decade=True, name="energy_true")
migra_axis = MapAxis.from_bounds(0.5, 2, nbin=150, node_type="edges", name="migra")

# Pointing offset 0.5 deg from target center
source_coord = SkyCoord.from_name("NGC1068").icrs
offset = 0.5 * u.deg
pointing = source_coord.directional_offset_by(position_angle=0 * u.deg, separation=offset)

geom = WcsGeom.create(
    skydir=pointing, width=(10, 10), binsz=0.02,
    frame="icrs", axes=[energy_axis]
)

# ------------------ Load Instrument Response Functions ------------------
irfs = load_irf_dict_from_file(IRF_FILENAME)

# ------------------ Create Base Dataset ------------------
empty_dataset = MapDataset.create(
    geom=geom, energy_axis_true=energy_axis_true, migra_axis=migra_axis, name="my-dataset"
)

observation = Observation.create(
    obs_id="0001", pointing=pointing, livetime=LIVETIME, irfs=irfs
)

maker = MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"])
dataset = maker.run(empty_dataset, observation)
dataset_path = BASE_PATH / f"dataset_NGC1068_{int(LIVETIME.value)}hr.fits"
dataset.write(dataset_path, overwrite=True)

# ------------------ Define Source & Background Models ------------------
spatial_model = PointSpatialModel(
    lon_0=source_coord.ra, lat_0=source_coord.dec, frame="icrs"
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
models.write(BASE_PATH / f"{SOURCE_NAME_AN}.yaml", overwrite=True)

# Re-load dataset and attach models (RAM-safe)
dataset = MapDataset.read(dataset_path, name="my-dataset")
dataset.models = models

# ------------------ Simulate Event Lists ------------------
print(f"\033[96m Simulating observations \033[0m")
for i in range(Nsim):
    
    obs = Observation.create(
        obs_id=i,
        pointing=pointing,
        livetime=LIVETIME,
        irfs=irfs,
        tstart=30 * u.min,
        reference_time=Time("2026-05-28T00:00:00", format="isot", scale="utc"),
        location=observatory_locations["cta_north"]
    )

    sampler = MapDatasetEventSampler(random_state=i)
    events = sampler.run(dataset, obs)
    obs.events = events

    # Save simulated events
    event_path = BASE_PATH / f"{Nsim}sims/events/{SOURCE_NAME_AN}_{int(LIVETIME.value)}hr_events_{i}.fits"
    obs.write(event_path, include_irfs=False, overwrite=True)
