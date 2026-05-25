"""Asset catalog, ControlNet models, renderer capabilities, video capabilities."""
from __future__ import annotations

from fastapi import APIRouter

from ..assets import list_assets
from ..renderers import installed_runtime_versions, renderer_capabilities
from ..schemas import AssetCatalog

router = APIRouter()


@router.get("/assets")
def assets() -> AssetCatalog:
    return AssetCatalog.model_validate(list_assets())


@router.get("/controlnet/models")
def controlnet_models() -> list[dict]:
    from ..controlnet import list_local_controlnet_models
    return list_local_controlnet_models()


@router.get("/renderers/capabilities")
def renderers_capabilities() -> dict[str, object]:
    capabilities = renderer_capabilities()
    capabilities["versions"] = installed_runtime_versions()
    return capabilities


@router.get("/video/capabilities")
def video_capabilities() -> dict[str, object]:
    return {"models": {}}
