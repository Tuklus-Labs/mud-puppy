# Heretic subprocess integration (parked)

These files were the first attempt at pipeline integration, calling the
upstream p-e-w/heretic-llm tool as a subprocess. Retired in favor of a
native ROCm-clean implementation at `mud_puppy/abliterate.py`, which:

- uses mud-puppy's own torch-ROCm stack (no CUDA-built venv)
- handles Mistral3 VLM + gpt-oss MoE via `load_model_graceful`
- has no interactive menus to monkey-patch
- applies the same Arditi et al. 2024 algorithm heretic wraps

Kept here for reference in case Gary ever wants to compare against heretic
output or resurrect this path.
