"""API package — FastAPI routers split by feature domain.

Sub-modules
-----------
state    — shared mutable state, engine registry, task helpers
system   — /health, /system/tasks, /system/gpus
assets   — /assets, /controlnet/models, /renderers/capabilities, /video/capabilities
sources  — /sources, /sources/pick-root, /sources/image
layer    — /layer/*, /ws/tasks/*
inpaint  — /ws/inpaint
"""
