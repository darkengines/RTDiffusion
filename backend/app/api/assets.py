"""Asset catalog, ControlNet models, renderer capabilities, video capabilities."""
from __future__ import annotations

from fastapi import APIRouter

from ..assets import list_assets
from ..renderers import installed_runtime_versions, renderer_capabilities
from ..motion import missing_motion_adapter_message, motion_adapter_configured
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
    models = ["causal-forcing", "causal-forcing-2step", "causal-forcing-1step", "krea-realtime-video", "fastvideo"]
    return {
        "models": {
            model: {
                "configured": motion_adapter_configured(model),
                "message": "" if motion_adapter_configured(model) else missing_motion_adapter_message(model),
            }
            for model in models
        }
    }
