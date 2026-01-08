"""Verified dataset sources for the GAIA pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import xarray as xr


DEFAULT_CDS_ENV_VARS = ("CDSAPI_URL", "CDSAPI_KEY")


class CredentialError(RuntimeError):
    """Raised when dataset credentials are missing."""


def verify_access_credentials(
    env_vars: Iterable[str] = DEFAULT_CDS_ENV_VARS,
    config_paths: Iterable[Path | str] = (Path.home() / ".cdsapirc",),
) -> None:
    """Ensure credentials exist before accessing protected datasets.

    Args:
        env_vars: Environment variables that should be populated.
        config_paths: Paths to check for credential configuration.

    Raises:
        CredentialError: If no credentials are found.
    """
    env_missing = [var for var in env_vars if not _get_env(var)]
    config_missing = [Path(path) for path in config_paths if not Path(path).exists()]
    if env_missing and len(config_missing) == len(list(config_paths)):
        env_display = ", ".join(env_missing)
        config_display = ", ".join(str(path) for path in config_missing)
        raise CredentialError(
            "Missing dataset credentials. Configure environment variables "
            f"({env_display}) or credential files ({config_display})."
        )


def _get_env(name: str) -> str | None:
    """Return an environment variable value if set."""
    try:
        from os import getenv

        return getenv(name)
    except Exception:
        return None


@dataclass(frozen=True)
class ERA5ZarrSource:
    """Access ERA5 data stored in a Zarr-compatible object store."""

    store_url: str
    consolidated: bool = True
    storage_options: Mapping[str, str] | None = None

    def open_dataset(self) -> xr.Dataset:
        """Open the dataset after confirming access credentials."""
        verify_access_credentials()
        return xr.open_zarr(
            self.store_url,
            consolidated=self.consolidated,
            storage_options=self.storage_options,
        )
