from pydantic import BaseModel, Field


class LayerCondition(BaseModel):
    name: str = Field(default="", max_length=200)
    prompt: str = Field(default="", max_length=2000)
    negative_prompt: str = Field(default="", max_length=2000)
    model_path: str | None = None
    lora_paths: list[str] = Field(default_factory=list)
    image: str
    weight: float = Field(default=1.0, ge=0.0, le=4.0)
    mode: str = Field(default="mask", max_length=32)
    cfg: float | None = Field(default=None, ge=0, le=30)
    steps: int | None = Field(default=None, ge=1, le=40)
    sampler: str | None = Field(default=None, max_length=80)
    scheduler: str | None = Field(default=None, max_length=80)
    denoise: float | None = Field(default=None, ge=0.0, lt=1.0)
    schedule: str = Field(default="auto", max_length=32)
    schedule_start: float = Field(default=0.0, ge=0.0, le=1.0)
    schedule_end: float = Field(default=1.0, ge=0.0, le=1.0)


class InpaintFrame(BaseModel):
    prompt: str = Field(default="", max_length=2000)
    negative_prompt: str = Field(default="", max_length=2000)
    model_path: str | None = Field(default=None, max_length=1000)
    lora_paths: list[str] = Field(default_factory=list, max_length=16)
    image: str
    mask: str
    width: int = Field(default=512, ge=256, le=1536)
    height: int = Field(default=512, ge=256, le=1536)
    steps: int = Field(default=1, ge=1, le=32)
    strength: float = Field(default=0.38, ge=0.0, lt=1.0)
    cfg: float = Field(default=1.5, ge=0.0, le=30.0)
    sampler: str | None = Field(default=None, max_length=64)
    scheduler: str | None = Field(default="simple", max_length=64)
    layer_conditions: list[LayerCondition] = Field(default_factory=list, max_length=16)


class InpaintResult(BaseModel):
    image: str
    fps: float
    latency_ms: float
    mode: str
    error: str | None = None


class LayerGenerateFrame(BaseModel):
    prompt: str = Field(default="", max_length=2000)
    negative_prompt: str = Field(default="", max_length=2000)
    model_path: str | None = Field(default=None, max_length=1000)
    lora_paths: list[str] = Field(default_factory=list, max_length=16)
    width: int = Field(default=832, ge=256, le=1536)
    height: int = Field(default=1216, ge=256, le=1536)
    variations: int = Field(default=4, ge=1, le=12)
    steps: int = Field(default=8, ge=1, le=40)
    cfg: float = Field(default=4.2, ge=0.0, le=30.0)
    sampler: str | None = Field(default=None, max_length=64)
    scheduler: str | None = Field(default="simple", max_length=64)
    transparent_background: bool = True
    seed: int | None = Field(default=None, ge=0)


class LayerVariation(BaseModel):
    image: str
    seed: int


class LayerGenerateResult(BaseModel):
    variations: list[LayerVariation]
    latency_ms: float
    mode: str
    error: str | None = None


class LayerTaskStart(BaseModel):
    task_id: str


class LayerTaskProgress(BaseModel):
    task_id: str
    status: str
    phase: str
    progress: float = Field(default=0, ge=0, le=1)
    message: str = ""
    result: LayerGenerateResult | None = None
    error: str | None = None


class AssetItem(BaseModel):
    name: str
    path: str
    size: int
    preferred: bool = False


class AssetCatalog(BaseModel):
    models: list[AssetItem]
    loras: list[AssetItem]