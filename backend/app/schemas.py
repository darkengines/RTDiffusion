from pydantic import BaseModel, Field


class ControlNetPreprocessorParams(BaseModel):
    canny_low: int = Field(default=100, ge=0, le=255)
    canny_high: int = Field(default=200, ge=0, le=255)
    openpose_hands: bool = True
    openpose_face: bool = True
    mlsd_thr_v: float = Field(default=0.1, ge=0.0, le=10.0)
    mlsd_thr_d: float = Field(default=0.1, ge=0.0, le=10.0)
    lineart_coarse: bool = False


class LayerCondition(BaseModel):
    name: str = Field(default="", max_length=200)
    layer_id: str = Field(default="", max_length=200)
    z_index: float | None = None
    region_id: str = Field(default="", max_length=200)
    prompt: str = Field(default="", max_length=2000)
    negative_prompt: str = Field(default="", max_length=2000)
    model_path: str | None = None
    lora_paths: list[str] = Field(default_factory=list)
    image: str
    # Per-channel masks (see ``docs/LAYER_SYSTEM.md`` §3). All optional —
    # when omitted, the channel falls back to the alpha of ``image``. Each
    # value is a base64-encoded grayscale PNG data URL.
    denoise_mask: str | None = None
    prompt_mask: str | None = None
    cfg_mask: str | None = None
    color_mask: str | None = None
    color_image: str | None = None  # RGB pixels for color compositing (§5)
    # Binary-blob references (preferred for high-bandwidth channels). The
    # realtime frontend streams raw PNG bytes over the WebRTC ``resources``
    # data channel and stores the returned ID here. Backend resolves the ID
    # against the session's blob store. Bypasses the ~33% base64 overhead.
    denoise_mask_ref: str | None = Field(default=None, max_length=64)
    prompt_mask_ref: str | None = Field(default=None, max_length=64)
    cfg_mask_ref: str | None = Field(default=None, max_length=64)
    color_mask_ref: str | None = Field(default=None, max_length=64)
    weight: float = Field(default=1.0, ge=0.0, le=4.0)
    mode: str = Field(default="mask", max_length=32)
    mask_operator: str = Field(default="add", max_length=32)
    denoise_operator: str = Field(default="replace", max_length=32)
    cfg_operator: str = Field(default="replace", max_length=32)
    prompt_operator: str = Field(default="concat", max_length=32)
    negative_prompt_operator: str = Field(default="concat", max_length=32)
    cfg: float | None = Field(default=None, ge=0, le=30)
    steps: int | None = Field(default=None, ge=1, le=40)
    sampler: str | None = Field(default=None, max_length=80)
    scheduler: str | None = Field(default=None, max_length=80)
    denoise: float | None = Field(default=None, ge=0.0, le=1.0)
    schedule: str = Field(default="auto", max_length=32)
    schedule_start: float = Field(default=0.0, ge=0.0, le=1.0)
    schedule_end: float = Field(default=1.0, ge=0.0, le=1.0)
    blending_enabled: bool = True
    blending_radius: int = Field(default=32, ge=-512, le=512)
    blending_strength: float = Field(default=1.0, ge=0.0, le=4.0)
    # Auto-tagger
    auto_tag: bool = False
    auto_tag_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    auto_tag_refresh_frames: int = Field(default=5, ge=1, le=60)
    # ControlNet conditioning
    controlnet_model: str | None = Field(default=None, max_length=64)
    controlnet_model_path: str | None = Field(default=None, max_length=1000)
    controlnet_scale: float = Field(default=1.0, ge=0.0, le=2.0)
    controlnet_start_at: float = Field(default=0.0, ge=0.0, le=1.0)
    controlnet_end_at: float = Field(default=1.0, ge=0.0, le=1.0)
    controlnet_image: str | None = None
    controlnet_preprocessor: bool = False
    controlnet_preprocessor_params: ControlNetPreprocessorParams = Field(default_factory=ControlNetPreprocessorParams)
    controlnet_use_layer_frame: bool = False


class PipelineNode(BaseModel):
    """A single node in the per-session processing graph."""
    id: str = Field(max_length=200)
    type: str = Field(max_length=32)  # "tagger" | "cn_from_layer"
    layer_id: str = Field(default="", max_length=200)
    # tagger node
    tagger_model: str = Field(default="wd-eva02-large-v3", max_length=80)
    tagger_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    tagger_refresh_frames: int = Field(default=5, ge=1, le=60)
    # cn_from_layer node
    cn_model: str = Field(default="canny", max_length=64)
    cn_model_path: str | None = Field(default=None, max_length=1000)
    cn_scale: float = Field(default=1.0, ge=0.0, le=2.0)
    cn_start: float = Field(default=0.0, ge=0.0, le=1.0)
    cn_end: float = Field(default=1.0, ge=0.0, le=1.0)
    cn_preprocessor_params: ControlNetPreprocessorParams = Field(default_factory=ControlNetPreprocessorParams)


class InpaintFrame(BaseModel):
    client_frame_id: int = Field(default=0, ge=0)
    client_input_id: int = Field(default=0, ge=0)
    scene_id: str = Field(default="", max_length=128)
    prompt: str = Field(default="", max_length=2000)
    negative_prompt: str = Field(default="", max_length=2000)
    debug_streams: bool = False
    session_directory: str = Field(default="", max_length=2000)
    model_path: str | None = Field(default=None, max_length=1000)
    device: str | None = Field(default=None, max_length=32)
    lora_paths: list[str] = Field(default_factory=list, max_length=16)
    image: str
    mask: str
    width: int = Field(default=512, ge=256, le=1536)
    height: int = Field(default=512, ge=256, le=1536)
    steps: int = Field(default=1, ge=1, le=128)
    strength: float = Field(default=0.38, ge=0.0, le=1.0)
    cfg: float = Field(default=1.5, ge=0.0, le=30.0)
    sampler: str | None = Field(default=None, max_length=64)
    scheduler: str | None = Field(default="simple", max_length=64)
    residual_cfg: bool = False
    stochastic_similarity_filter: bool = False
    seed: int | None = Field(default=None, ge=0, le=4_294_967_295)
    seed_mode: str = Field(default="fixed", max_length=16)
    seed_variation: int = Field(default=0, ge=0, le=1_000_000)
    reuse_previous_latent: bool = False
    latent_reuse_denoise: float = Field(default=0.24, ge=0.0, lt=1.0)
    latent_reuse_noise: float = Field(default=0.0, ge=0.0, le=1.0)
    latent_reuse_noise_mode: str = Field(default="none", max_length=16)
    latent_reuse_restart: bool = False
    stochastic_blur: float = Field(default=0.0, ge=0.0, le=32.0)
    stream_diffusion: bool = False
    stream_direct: bool = False
    stream_timestep_indices: list[int] = Field(default_factory=lambda: [0, 16, 32, 45], max_length=16)
    stream_frame_buffer_size: int = Field(default=1, ge=1, le=4)
    stream_max_passes: int = Field(default=16, ge=1, le=512)
    stream_cfg_type: str = Field(default="self", max_length=16)
    stream_triton_compile: bool = False
    # When set (0..1), overrides stream_timestep_indices via an auto-picker:
    # higher quality → more denoising steps. None = use raw indices above.
    stream_quality: float | None = Field(default=None, ge=0.0, le=1.0)
    prompt_b: str = Field(default="", max_length=4096)
    prompt_lerp: float = Field(default=0.0, ge=0.0, le=1.0)
    transparent_background: bool = False
    transparent_background_mode: str = Field(default="border", max_length=16)
    transparent_background_color: str = Field(default="#ffffff", max_length=32)
    transparent_background_tolerance: float = Field(default=34.0, ge=0.0, le=255.0)
    transparent_alpha_blur: float = Field(default=1.5, ge=0.0, le=24.0)
    transparent_alpha_threshold: int = Field(default=10, ge=0, le=255)
    layer_conditions: list[LayerCondition] = Field(default_factory=list, max_length=16)
    render_strategy: str = Field(default="single", max_length=32)
    tile_divisions: int = Field(default=2, ge=1, le=8)
    tile_overlap: int = Field(default=128, ge=0, le=512)
    layer_bbox_padding: int = Field(default=64, ge=0, le=256)

    def allows_scene_result_reuse(self) -> bool:
        """Whether transports may replay a cached scene result for this frame.

        This is part of the transport-facing contract rather than a pipeline-
        specific detail. Debug streams require a fresh backend result so image
        channels are regenerated consistently across renderers.
        """
        return (not self.stream_diffusion) and (not self.debug_streams) and bool(self.scene_id)


class InpaintResult(BaseModel):
    client_frame_id: int = 0
    client_input_id: int = 0
    scene_id: str = ""
    image: str
    fps: float
    latency_ms: float
    mode: str
    error: str | None = None
    debug_channels: dict[str, str] = Field(default_factory=dict)


class LayerGenerateFrame(BaseModel):
    prompt: str = Field(default="", max_length=2000)
    negative_prompt: str = Field(default="", max_length=2000)
    model_path: str | None = Field(default=None, max_length=1000)
    device: str | None = Field(default=None, max_length=32)
    lora_paths: list[str] = Field(default_factory=list, max_length=16)
    width: int = Field(default=832, ge=256, le=1536)
    height: int = Field(default=1216, ge=256, le=1536)
    variations: int = Field(default=4, ge=1, le=12)
    steps: int = Field(default=8, ge=1, le=40)
    cfg: float = Field(default=4.2, ge=0.0, le=30.0)
    sampler: str | None = Field(default=None, max_length=64)
    scheduler: str | None = Field(default="simple", max_length=64)
    transparent_background: bool = True
    transparent_background_method: str = Field(default="auto", max_length=16)
    transparent_background_mode: str = Field(default="border", max_length=16)
    transparent_background_color: str = Field(default="#ffffff", max_length=32)
    transparent_background_tolerance: float = Field(default=34.0, ge=0.0, le=255.0)
    transparent_alpha_blur: float = Field(default=1.5, ge=0.0, le=24.0)
    transparent_alpha_threshold: int = Field(default=10, ge=0, le=255)
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


class MotionClipRequest(BaseModel):
    name: str = Field(default="", max_length=200)
    source_image: str
    arrival_image: str = ""
    prompt: str = Field(default="", max_length=2000)
    arrival_prompt: str = Field(default="", max_length=2000)
    negative_prompt: str = Field(default="", max_length=2000)
    model_path: str | None = Field(default=None, max_length=1000)
    device: str | None = Field(default=None, max_length=32)
    lora_paths: list[str] = Field(default_factory=list, max_length=16)
    model: str = Field(default="krea-realtime-video", max_length=80)
    motion: str = Field(default="idle", max_length=80)
    region: str = Field(default="full", max_length=80)
    fps: int = Field(default=16, ge=4, le=60)
    duration_seconds: float = Field(default=2.0, ge=0.25, le=12.0)
    loop: str = Field(default="pingpong", max_length=32)
    intensity: float = Field(default=1.0, ge=0.0, le=3.0)
    width: int = Field(default=512, ge=128, le=1536)
    height: int = Field(default=512, ge=128, le=1536)
    arrival_steps: int = Field(default=12, ge=1, le=40)
    arrival_strength: float = Field(default=0.42, ge=0.0, lt=1.0)
    arrival_cfg: float = Field(default=4.5, ge=0.0, le=30.0)
    arrival_sampler: str | None = Field(default=None, max_length=64)
    arrival_scheduler: str | None = Field(default="simple", max_length=64)


class MotionClipFrame(BaseModel):
    index: int
    url: str
    duration_ms: int


class MotionClip(BaseModel):
    id: str
    name: str
    motion: str
    region: str
    model: str
    prompt: str = ""
    arrival_prompt: str = ""
    negative_prompt: str = ""
    keyframes: list[str] = Field(default_factory=list)
    fps: int
    duration_seconds: float
    loop: str
    width: int
    height: int
    frame_count: int
    frames: list[MotionClipFrame]
    video_url: str = ""
    created_at: float


class MotionTaskStart(BaseModel):
    task_id: str


class MotionTaskProgress(BaseModel):
    task_id: str
    status: str
    phase: str
    progress: float = Field(default=0, ge=0, le=1)
    message: str = ""
    preview_frames: list[MotionClipFrame] = Field(default_factory=list)
    result: MotionClip | None = None
    error: str | None = None


class AssetItem(BaseModel):
    name: str
    path: str
    size: int
    preferred: bool = False


class AssetCatalog(BaseModel):
    models: list[AssetItem]
    video_models: list[AssetItem] = Field(default_factory=list)
    loras: list[AssetItem]