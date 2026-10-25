"""Whole-field member availability without imputing missing realizations."""
from __future__ import annotations

import numpy as np
import xarray as xr


def member_availability(field: xr.DataArray) -> xr.DataArray:
    """Identify member-years with at least one finite spatial cell."""
    return np.isfinite(field).any(dim=("lat", "lon"))


def available_member_mean(field: xr.DataArray) -> xr.DataArray:
    """Average available whole fields, preserving missing cells inside a field."""
    available = member_availability(field)
    count = available.sum("member")
    if bool((count == 0).any()):
        raise ValueError("No available target members in at least one year")
    # Zero contributes only to the sum for an absent whole field, never as an
    # imputed realization or as a replacement for a missing cell in a present field.
    return field.where(available, 0).sum("member", skipna=False) / count


def complete_period_members(field: xr.DataArray) -> xr.DataArray:
    """Keep whole trajectories present throughout the supplied scoring period."""
    keep = member_availability(field).all("year")
    selected = field.sel(member=field.member.values[keep.values])
    if selected.sizes["member"] == 0:
        raise ValueError("No complete member trajectory over the evaluation period")
    return selected
