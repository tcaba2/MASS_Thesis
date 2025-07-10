# === IMPORTS ===
import numpy as np
import matplotlib.pyplot as plt
import copy
from pathlib import Path

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table

from gammapy.data import EventList
from gammapy.datasets import MapDataset, FluxPointsDataset
from gammapy.maps import MapAxis, WcsGeom, Map
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    Models,
    SkyModel,
    FoVBackgroundModel,
    ExpCutoffPowerLawSpectralModel,
    PointSpatialModel,
)
from gammapy.estimators import FluxPointsEstimator

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# === CONFIGURATION ===
BASE_PATH = Path("/Users/tharacaba/Desktop/Tesis_2/MASS_Thesis/simulations/Fermi")
MainSource = "NGC1068"
MainSourceAn = "NGC1068_Fermi"
Nsim = 5
exposures = [50]
param_names = ["index", "amplitude", "lambda_"]

spectral_model = ExpCutoffPowerLawSpectralModel(
    amplitude=6.35e-14 * u.Unit("cm-2 s-1 TeV-1"),
    index=2.3,
    lambda_=0.1 * u.Unit("TeV-1"),
    reference=1 * u.TeV,
)

# === GEOMETRY & MODELS ===

e_min, e_max, e_bins = 0.1, 100.0, 10
e_edges = np.logspace(np.log10(e_min), np.log10(e_max), e_bins) * u.TeV

center = SkyCoord.from_name(MainSource).icrs
offset = 0.5 * u.deg
pointing_position = center.directional_offset_by(position_angle=0 * u.deg, separation=offset)

ENERGY_AXIS = MapAxis.from_energy_bounds("0.1 TeV", "100.0 TeV", nbin=10, per_decade=True)
WCS_GEOM = WcsGeom.create(skydir=pointing_position, width=(10, 10), binsz=0.02, frame="icrs", axes=[ENERGY_AXIS])

spatial_model = PointSpatialModel(lon_0="40.669 deg", lat_0="-0.013 deg", frame="icrs")
sky_model = SkyModel(spectral_model=spectral_model, spatial_model=spatial_model, name=MainSourceAn)
sky_model.spatial_model.parameters["lon_0"].frozen = True
sky_model.spatial_model.parameters["lat_0"].frozen = True
bkg_model = FoVBackgroundModel(dataset_name="my-dataset")
models = Models([sky_model, bkg_model])

# === UTILITY ===
def save_figure(filename):
    path = BASE_PATH / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def gaussian(x, norm, mean, sigma):
    return norm * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

def fit_all_simulations(ext):
    print(f"\n--- Fitting for {ext} hr exposure ---")
    for i in range(Nsim):
        dataset = MapDataset.read(BASE_PATH / f'dataset_{MainSource}_{ext}hr.fits')
        dataset.models = copy.deepcopy(models)

        events = EventList.read(BASE_PATH / f"{Nsim}sims/events/{MainSource}_{ext}hr_events_{i}.fits")
        counts = Map.from_geom(WCS_GEOM)
        counts.fill_events(events)
        dataset.counts = counts

        print(f"  Fit simulation {i}")
        fit = Fit(optimize_opts={"print_level": 1})
        _ = fit.run([dataset])
        dataset.models.write(BASE_PATH / f"{Nsim}sims/best-fit/{MainSource}_{ext}hr_best-fit_{i}.yaml", overwrite=True)

def collect_fit_parameters(ext):
    print(f"\n--- Collecting best-fit parameters for {ext} hr ---")
    result_dir = BASE_PATH / f"{Nsim}sims/best-fit"
    rows = []

    for filepath in result_dir.glob(f"{MainSource}*{ext}hr*best-fit*.yaml"):
        model = Models.read(filepath)
        row = {f"{p.name}": p.value for p in model[MainSourceAn].spectral_model.parameters}
        row.update({f"{p.name}_err": p.error for p in model[MainSourceAn].spectral_model.parameters})
        rows.append(row)

    result_table = Table(rows)
    result_table.write(result_dir / f"{MainSource}_{ext}hr_results.fits", overwrite=True)
    return result_table

def plot_histograms(result_table, ext):
    Res_mean, Res_sigma = [], []
    for name in param_names:
        values = result_table[name]
        mean, sigma = np.mean(values), np.std(values)
        Res_mean.append(mean)
        Res_sigma.append(sigma)

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        n, bins, _ = axs[0].hist(values, bins=15, density=True)
        x = np.linspace(mean - 3*sigma, mean + 3*sigma, 100)
        axs[1].plot(x, gaussian(x, np.max(n), mean, sigma), label=f"{mean:.3g} ± {sigma:.3g}")
        axs[1].legend()
        axs[0].set_title(f"{name} Histogram")
        axs[1].set_title(f"Gaussian Fit: {name}")
        save_figure(f"{Nsim}sims/{MainSource}_{name}_{ext}hr_hist.png")

    print("Mean values:", Res_mean)
    print("Standard deviations:", Res_sigma)
    return Res_mean, Res_sigma

def get_best_matching_sim(result_table, Res_mean):
    distances = np.sqrt(sum((result_table[p] - m) ** 2 for p, m in zip(param_names, Res_mean)))
    best_idx = int(np.argmin(distances))
    print(f"\n>>> Best matching simulation index: {best_idx}")
    return best_idx

def compute_flux_points(best_idx, Res_mean, Res_sigma, ext):
    events_best = EventList.read(BASE_PATH / f"{Nsim}sims/events/{MainSource}_{ext}hr_events_{best_idx}.fits")
    counts_best = Map.from_geom(WCS_GEOM)
    counts_best.fill_events(events_best)

    dataset_best = MapDataset.read(BASE_PATH / f'dataset_{MainSource}_{ext}hr.fits')
    dataset_best.counts = counts_best
    model_best = Models.read(BASE_PATH / f"{Nsim}sims/best-fit/{MainSource}_{ext}hr_best-fit_{best_idx}.yaml")
    print(model_best)

    dataset_best.models = model_best
    for pname, mean, sigma in zip(param_names, Res_mean, Res_sigma):
        param = getattr(dataset_best.models[0].spectral_model, pname)
        param.value = mean
        param.error = sigma

    fpe = FluxPointsEstimator(
        energy_edges=e_edges,
        source=MainSourceAn,
        selection_optional="all",
        n_sigma=1,
        n_sigma_ul=3,
        n_jobs=7,
    )
    flux_points = fpe.run(datasets=dataset_best)
    flux_points.sqrt_ts_threshold_ul = 3
    flux_points.to_table(sed_type="dnde", formatted=True)
    return flux_points, dataset_best.models

def plot_flux_and_model(best_idx, flux_points, model, ext):
    dataset_original = Models.read(BASE_PATH / f"./{MainSourceAn}.yaml")
    ax = dataset_original[0].spectral_model.plot(
        energy_bounds=[e_min, e_max] * u.TeV,
        sed_type="e2dnde",
        label="Sim. model",
        color="black"
    )

    fp_dataset = FluxPointsDataset(data=flux_points, models=model[0])
    fp_dataset.plot_spectrum(ax=ax, kwargs_fp={"color": "red", "marker": "o"}, kwargs_model={"color": "blue"})
    ax.legend()
    save_figure(f"{MainSource}_flux_points_{ext}hr.png")
    print(fp_dataset)

# === MAIN PIPELINE ===
def main():
    for ext in exposures:
        fit_all_simulations(ext)
        result_table = collect_fit_parameters(ext)
        Res_mean, Res_sigma = plot_histograms(result_table, ext)
        best_idx = get_best_matching_sim(result_table, Res_mean)
        flux_points, model = compute_flux_points(best_idx, Res_mean, Res_sigma, ext)
        plot_flux_and_model(best_idx, flux_points, model, ext)

if __name__ == "__main__":
    main()
