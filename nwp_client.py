"""
Client for the Earth-2 FourCastNet NIM (see the OpenAPI spec).

Role in VAJRA: supplies the NWP environment channels (moisture, temperature,
winds) for the fusion pipeline. It is NOT the nowcast model itself: it is
global, ~25 km, multi-hour steps. Regrid its output to the 128x128 analysis
grid and hold it constant between model steps.

pip install requests numpy
"""

import io
import tarfile
from datetime import datetime, timezone
from typing import Callable, Mapping, Sequence

import numpy as np
import requests

BASE_URL = "http://localhost:8000"  # where your NIM container is exposed


def is_ready(base_url: str = BASE_URL) -> bool:
    try:
        return requests.get(f"{base_url}/v1/health/ready", timeout=5).status_code == 200
    except requests.RequestException:
        return False


def forecast(
    initial_state: np.ndarray,
    start_time: datetime,
    steps: int = 4,
    variables: str = "t2m,tcwv,u10m,v10m",
    seed: int = 0,
    base_url: str = BASE_URL,
) -> dict:
    """
    initial_state: float32 array shaped (batch, 1, variables, lat, lon), built
                   from ERA5/GFS analysis. Variable order/count must match the
                   model (see NIM docs).
    Returns {(lead_step, batch_index): ndarray}.
    """
    buf = io.BytesIO()
    np.save(buf, initial_state.astype(np.float32))
    buf.seek(0)

    resp = requests.post(
        f"{base_url}/v1/infer",
        files={"input_array": ("input.npy", buf, "application/octet-stream")},
        data={
            "input_time": start_time.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "simulation_length": steps,
            "variables": variables,
            "seed": seed,
        },
        timeout=600,
    )
    resp.raise_for_status()

    out = {}
    with tarfile.open(fileobj=io.BytesIO(resp.content)) as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".npy"):
                continue
            lead, batch = member.name.removesuffix(".npy").split("_")  # <lead>_<batch>.npy
            out[(int(lead), int(batch))] = np.load(io.BytesIO(tar.extractfile(member).read()))
    return out


def regrid_to_analysis_grid(field: np.ndarray, lat_slice: slice, lon_slice: slice,
                            out_hw=(128, 128)) -> np.ndarray:
    """Crop the global field to your region, then bilinearly resample."""
    from scipy.ndimage import zoom
    region = field[lat_slice, lon_slice]
    return zoom(region, (out_hw[0] / region.shape[0], out_hw[1] / region.shape[1]), order=1)


def make_nwp_loader(
    forecast_fields: Mapping[tuple[int, int], np.ndarray],
    variables: Sequence[str],
    lat_slice: slice,
    lon_slice: slice,
    cape_loader: Callable | None = None,
    lead_step: int = 0,
    batch_index: int = 0,
    pwat_variable: str = "tcwv",
    cape_variable: str | None = None,
    out_hw: tuple[int, int] = (128, 128),
) -> Callable:
    """Adapt one FourCastNet output into the pipeline's ``(CAPE, PWAT)`` grids.

    ``variables`` must match the order sent to :func:`forecast`. FourCastNet
    commonly provides ``tcwv`` (total-column water vapor), while CAPE may need
    to come from a separate NWP source via ``cape_loader(timestamp)``.
    """
    variable_indices = {name: index for index, name in enumerate(variables)}
    if pwat_variable not in variable_indices:
        raise ValueError(f"{pwat_variable!r} is not present in variables")
    if cape_variable is not None and cape_variable not in variable_indices:
        raise ValueError(f"{cape_variable!r} is not present in variables")
    if cape_variable is None and cape_loader is None:
        raise ValueError("Provide cape_variable or a separate cape_loader")

    key = (lead_step, batch_index)
    if key not in forecast_fields:
        raise ValueError(f"FourCastNet output does not contain lead/batch {key}")
    fields = np.asarray(forecast_fields[key])
    if fields.ndim != 3 or fields.shape[0] != len(variables):
        raise ValueError(
            "Expected output shape (variables, latitudes, longitudes) matching variables"
        )

    def select_field(variable: str) -> np.ndarray:
        return regrid_to_analysis_grid(
            fields[variable_indices[variable]], lat_slice, lon_slice, out_hw
        ).astype(np.float32)

    def load_nwp(timestamp) -> tuple[np.ndarray, np.ndarray]:
        cape = (
            select_field(cape_variable)
            if cape_variable is not None
            else np.asarray(cape_loader(timestamp), dtype=np.float32)
        )
        pwat = select_field(pwat_variable)
        if cape.shape != out_hw or pwat.shape != out_hw:
            raise ValueError(f"NWP fields must both have shape {out_hw}")
        return cape, pwat

    return load_nwp


if __name__ == "__main__":
    print("NIM ready:", is_ready())
