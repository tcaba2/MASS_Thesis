# === IMPORTS ===
# Standard library 
import copy
import gc
import warnings
from pathlib import Path
import os

# Third-pary libraries 
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

# Gammapy
from gammapy.data import EventList
from gammapy.datasets import MapDataset, FluxPointsDataset
from gammapy.estimators import FluxPointsEstimator, FluxPoints
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    ExpCutoffPowerLawSpectralModel,
    FoVBackgroundModel,
    Models,
    PointSpatialModel,
    SkyModel,
    BrokenPowerLawSpectralModel,
    SuperExpCutoffPowerLaw3FGLSpectralModel,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# === CONFIGURATION ===
BASE_PATH = Path("/Users/tharacaba/Desktop/Tesis_2/MASS_Thesis/simulations/Wind")
MainSource = "NGC1068"
MainSourceAn = "NGC1068_Wind"
Nsim = 100
exposures = [100]
param_names = ["index_1", "index_2", "amplitude", "ecut"]

# === MODEL SETUP ===
spectral_model = SuperExpCutoffPowerLaw3FGLSpectralModel(
    index_1 = 2,
    index_2 = 2,
    amplitude = "1e-14 TeV-1 s-1 cm-2",
    reference = "1 TeV",
    ecut = "0.5 TeV",
)

center = SkyCoord.from_name(MainSource).icrs
offset = 0.5 * u.deg
pointing_position = center.directional_offset_by(position_angle=0 * u.deg, separation=offset)

ENERGY_AXIS = MapAxis.from_energy_bounds("0.1 TeV", "100.0 TeV", nbin=10, per_decade=True)
WCS_GEOM = WcsGeom.create(skydir=pointing_position, width=(10, 10), binsz=0.02, frame="icrs", axes=[ENERGY_AXIS])

spatial_model = PointSpatialModel(lon_0=center.ra, lat_0=center.dec, frame="icrs")
sky_model = SkyModel(spectral_model=spectral_model, spatial_model=spatial_model, name=MainSourceAn)
sky_model.spatial_model.parameters["lon_0"].frozen = True
sky_model.spatial_model.parameters["lat_0"].frozen = True
#sky_model.spectral_model.parameters["alpha"].frozen = False

bkg_model = FoVBackgroundModel(dataset_name="my-dataset")
models = Models([sky_model, bkg_model])

spectral_model.amplitude.min = 1e-17
spectral_model.amplitude.max = 1e-11
spectral_model.index_1.min = 1
spectral_model.index_1.max = 3.5
spectral_model.index_2.min = 0.5
spectral_model.index_2.max = 5
spectral_model.ecut.min = 0.1
spectral_model.ecut.max = 100

# === ENERGY GRID ===
e_min, e_max, e_bins = 0.1, 100.0, 10
e_edges = np.logspace(np.log10(e_min), np.log10(e_max), e_bins) * u.TeV

# === UTILITY ===
def print_cyan(msg):
    print(f"\033[96m{msg}\033[0m")

def save_figure(filename):
    path = BASE_PATH / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def gaussian(x, norm, mean, sigma):
    return norm * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

# === SIMULATION FITTING ===
def fit_all_simulations(ext):
    print_cyan(f"\n--- Fitting for {ext} hr exposure ---")
    for i in range(Nsim):
        dataset = MapDataset.read(BASE_PATH / f'dataset_{MainSource}_{ext}hr.fits')
        dataset.models = models.copy()

        events = EventList.read(BASE_PATH / f"{Nsim}sims/events/{MainSourceAn}_{ext}hr_events_{i}.fits")
        counts = Map.from_geom(WCS_GEOM)
        counts.fill_events(events)
        dataset.counts = counts

        print_cyan(f"  > Fit simulation {i}")
        fit = Fit(optimize_opts={"print_level": 1})
        _ = fit.run([dataset])

        dataset.models.write(BASE_PATH / f"{Nsim}sims/best-fit/{MainSource}_{ext}hr_best-fit_{i}.yaml", overwrite=True)

        del dataset, events, counts, fit
        gc.collect()

# === PARAMETER COLLECTION ===
def collect_fit_parameters(ext):
    print_cyan(f"\n--- Collecting best-fit parameters for {ext} hr ---")
    result_dir = BASE_PATH / f"{Nsim}sims/best-fit"
    rows = []

    for filepath in result_dir.glob(f"{MainSource}*{ext}hr*best-fit*.yaml"):
        model = Models.read(filepath)
        spec = model[MainSourceAn].spectral_model
        row = {f"{p.name}": p.value for p in spec.parameters}
        row.update({f"{p.name}_err": p.error for p in spec.parameters})
        rows.append(row)
        del model
        gc.collect()

    table = Table(rows)
    table.write(result_dir / f"{MainSource}_{ext}hr_results.fits", overwrite=True)
    return table

# === PLOT PARAMETER DISTRIBUTIONS ===
def plot_histograms(table, ext):
    Res_mean, Res_sigma = [], []

    for name in param_names:
        values = table[name]
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

    print_cyan("\nMean Values:")
    for name, val in zip(param_names, Res_mean):
        print(f"{name:>15}: {val:.4g}")
    
    print_cyan("\nStandard Deviations:")
    for name, val in zip(param_names, Res_sigma):
        print(f"{name:>15}: {val:.4g}")

    return Res_mean, Res_sigma

# === FLUX POINT AVERAGING ===
def average_flux_points(ext, Res_mean, Res_sigma):
    print_cyan(f"\n--- Computing averaged flux points for {ext} hr ---")

    flux_dir = BASE_PATH / f"{Nsim}sims/spectra"
    flux_dir.mkdir(parents=True, exist_ok=True)

    for i in range(Nsim):
        events = EventList.read(BASE_PATH / f"{Nsim}sims/events/{MainSourceAn}_{ext}hr_events_{i}.fits")
        counts = Map.from_geom(WCS_GEOM)
        counts.fill_events(events)

        dataset = MapDataset.read(BASE_PATH / f'dataset_{MainSource}_{ext}hr.fits')
        dataset.counts = counts
        model = Models.read(BASE_PATH / f"{Nsim}sims/best-fit/{MainSource}_{ext}hr_best-fit_{i}.yaml")
        dataset.models = model

        fpe = FluxPointsEstimator(
            energy_edges=e_edges,
            source=MainSourceAn,
            selection_optional="all",
            n_sigma=1,
            n_sigma_ul=3,
            n_jobs=1,
        )
        flux_points = fpe.run(datasets=dataset)
        flux_points.write(flux_dir / f"{MainSource}_{ext}hr_fp_{i}.fits", overwrite=True)

        del events, counts, dataset, model, flux_points
        gc.collect()

    # Aggregate saved flux point files
    flux_tables = []
    for i in range(Nsim):
        fp = FluxPoints.read(flux_dir / f"{MainSource}_{ext}hr_fp_{i}.fits")
        flux_tables.append(fp.to_table(sed_type="dnde", formatted=False))
        del fp
        gc.collect()

    n_bins = len(flux_tables[0])
    avg_table = flux_tables[0].copy()

    for i in range(n_bins):
        dndes = np.array([t["dnde"][i] for t in flux_tables], dtype=np.float32)
        ts_vals = np.array([t["ts"][i] for t in flux_tables], dtype=np.float32)
        ts_mean = np.nanmean(ts_vals)

        if ts_mean > 9:
            avg = np.nanmean(dndes)
            std = np.nanstd(dndes)
            avg_table["dnde"][i] = avg
            avg_table["dnde_errn"][i] = std
            avg_table["dnde_errp"][i] = std
            avg_table["is_ul"][i] = False
            avg_table["dnde_ul"][i] = np.nan
        else:
            avg_table["dnde_ul"][i] = np.nanpercentile(dndes, 95)
            avg_table["dnde"][i] = np.nan
            avg_table["dnde_errn"][i] = np.nan
            avg_table["dnde_errp"][i] = np.nan
            avg_table["is_ul"][i] = True

    model = Models.read(BASE_PATH / f"{Nsim}sims/best-fit/{MainSource}_{ext}hr_best-fit_1.yaml")
    for pname, mean, sigma in zip(param_names, Res_mean, Res_sigma):
        param = getattr(model[0].spectral_model, pname)
        param.value = mean
        param.error = sigma

    flux = FluxPoints.from_table(table=avg_table, reference_model=model[0])
    flux.write(BASE_PATH / f"{Nsim}sims/avg_flux_points_{ext}hr.fits", overwrite=True)
    return flux

# === FINAL PLOT ===
def plot_avg_flux_and_model(flux_points, ext, Res_mean, Res_sigma):
    dataset_original = Models.read(BASE_PATH / f"{MainSourceAn}.yaml")
    ax = dataset_original[0].spectral_model.plot(
        energy_bounds=[e_min, e_max] * u.TeV,
        sed_type="e2dnde",
        label="Sim. model",
        color="black"
    )

    model = Models.read(BASE_PATH / f"{Nsim}sims/best-fit/{MainSource}_{ext}hr_best-fit_1.yaml")
    for pname, mean, sigma in zip(param_names, Res_mean, Res_sigma):
        param = getattr(model[0].spectral_model, pname)
        param.value = mean
        param.error = sigma

    fp_dataset = FluxPointsDataset(data=flux_points, models=model[0])
    fp_dataset.plot_spectrum(ax=ax, kwargs_fp={"color": "red", "marker": "o"}, kwargs_model={"color": "blue"})
    ax.set_ylim(1e-17, 5e-12)
    #ax.set_xlim(0.01, 200)
    ax.legend()
    save_figure(f"{Nsim}sims/{MainSource}_avg_flux_points_{ext}hr.png")
    print(fp_dataset)

# === MAIN ===
def main():
    for ext in exposures:
        fit_all_simulations(ext)
        result_table = collect_fit_parameters(ext)
        Res_mean, Res_sigma = plot_histograms(result_table, ext)
        flux_points = average_flux_points(ext, Res_mean, Res_sigma)
        #flux_points = FluxPoints.read(BASE_PATH / f"{Nsim}sims/avg_flux_points_{ext}hr.fits")
        plot_avg_flux_and_model(flux_points, ext, Res_mean, Res_sigma)

    # Add the notification here
    os.system('osascript -e \'display notification "Python script done!" with title "VS Code"\'') 



if __name__ == "__main__":
    main()
