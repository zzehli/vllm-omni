# Native vLLM-Omni Support Plan for Boogu

## Context

Boogu-Image-0.1-Base is a diffusion-family image generation model. Its Hugging Face metadata identifies it as a Diffusers model, with a `BooguImagePipeline`, scheduler files, transformer diffusion weights, and a VAE. For vLLM-Omni, there are two levels of support:

- Adapter-level support: run the model through the generic Diffusers adapter with `--diffusion-load-format diffusers`.
- Native support: add a first-class vLLM-Omni implementation under `vllm_omni/diffusion/models`, register it, and enable vLLM-Omni features incrementally.

Native support should be built from output parity outward. First prove that the native implementation matches the upstream Diffusers pipeline, then add vLLM-Omni-specific performance features one at a time.

## Target Shape

Native support would likely add:

- `vllm_omni/diffusion/models/boogu_image/`
- `pipeline_boogu_image.py`
- `boogu_image_transformer.py`
- possibly a local scheduler file if Boogu's custom scheduler is not already available
- registry entries in `vllm_omni/diffusion/registry.py`
- post-processing functions
- examples, docs or recipe entry, and tests

## Step-by-Step Plan

| Step | Goal | Testable Result |
|---|---|---|
| 1. Baseline in plain Diffusers | Run Boogu exactly as intended upstream with fixed prompt, seed, size, steps, dtype. | A script saves `baseline.png` and records seed, config, revision, and latency. |
| 2. Baseline through vLLM-Omni Diffusers adapter | Try `--diffusion-load-format diffusers` before native code. | Either it serves an image, or you get a clear failure explaining missing custom pipeline support. |
| 3. Model inventory | Inspect `model_index.json`, `transformer/config.json`, `mllm/config.json`, scheduler, VAE config, and pipeline call signature. | A small manifest lists components: transformer, VAE, scheduler, MLLM/text processor, required inputs, and weight sources. |
| 4. Scaffold native module | Add `boogu_image` model folder, empty/native pipeline class, transformer class shell, and exports. | Import test passes: `from vllm_omni.diffusion.models.boogu_image import BooguImagePipeline`. |
| 5. Register pipeline | Add `"BooguImagePipeline"` to the diffusion registry and post-process registry. | `OmniDiffusionConfig(model="Boogu/...").enrich_config()` resolves to `BooguImagePipeline`. |
| 6. Load non-transformer components | Load scheduler, tokenizer/processor/MLLM pieces, and VAE from the HF repo. | Constructor test creates `BooguImagePipeline(od_config=...)` without loading transformer weights yet. |
| 7. Port transformer structure | Copy/adapt Boogu's `transformer_boogu.py` into native `nn.Module`, remove Diffusers mixins and training-only code. | Shape test: instantiate transformer from config and run one tiny/dummy forward through key blocks. |
| 8. Implement weight loading | Define `weights_sources` for `transformer`, maybe `vae`, maybe `mllm`, and implement `load_weights()`. | Weight-load test reports no unexpected missing parameters. |
| 9. Implement request-level `forward()` | Convert `OmniDiffusionRequest` into Boogu prompt encoding, denoising loop, scheduler steps, and VAE decode. | Offline inference script produces an image from one prompt. |
| 10. Add post-processing | Convert decoded tensor/latents to PIL or the configured output type. | Output type test returns valid PIL images through `DiffusionOutput`. |
| 11. Parity test vs Diffusers | Compare native output to baseline with the same seed/settings. | Image similarity is within an agreed tolerance, or known numerical differences are documented. |
| 12. Online serving | Add example server/client path. | `vllm serve ... --omni` accepts an OpenAI-compatible request and returns an image. |
| 13. Optimize attention | Replace compatible attention sites with vLLM-Omni `Attention` and assign roles like `self`/`cross`. | Output parity still passes; latency does not regress. |
| 14. Add optional advanced features | Sequence parallel, CFG parallel, CPU offload, step execution, cache acceleration only after basic parity. | Each feature has its own opt-in test and fallback behavior. |
| 15. Docs and recipe | Add recipe/example docs and supported-model entry if accepted. | A fresh user can run the documented command end to end. |

## Recommended Implementation Order

Start with steps 1 through 3 before touching native code. This creates the baseline and prevents guessing about Boogu's actual component graph.

Then implement steps 4 through 11 as the minimal native support path. Stop there until parity is solid.

Only after parity should steps 12 through 15 be added. Performance features such as vLLM-Omni attention, sequence parallelism, CFG parallelism, step execution, quantization, or cache acceleration should each land behind their own testable checkpoint.

## Completed: Adapter-Level Support (Steps 2–3)

Steps 2 and 3 are done. The model now serves images through `--diffusion-load-format diffusers`. The following changes were required.

### Environment setup

```bash
# Clone and install the Boogu companion package (contains BooguImagePipeline)
git clone https://github.com/boogu-project/Boogu-Image.git /root/dev/Boogu-Image
uv pip install --python /root/dev/.venv/bin/python --no-deps -e /root/dev/Boogu-Image/
uv pip install --python /root/dev/.venv/bin/python "torchao>=0.15,<0.18"
```

The `boogu-image` package must be installed because the HF model repo only ships re-export stubs (e.g. `transformer/transformer_boogu.py` imports from `boogu.models.transformers`). `BooguImagePipeline` itself is defined in `boogu.pipelines.boogu.pipeline_boogu` and is not part of the standard `diffusers` library.

### Code changes

**`vllm_omni/diffusion/models/diffusers_adapter/pipeline_diffusers_adapter.py`**

Three changes:

1. **Custom pipeline class discovery** (`_find_custom_pipeline_class` static method): When `diffusers_pipeline_cls` is `None` (class not in standard diffusers), the method reads import statements in the model's cached snapshot stubs to identify the owning Python package, then text-searches that package's source files for `class <ClassName>` without importing every submodule. Only the one matching file is imported. The resolved class is then used in place of `DiffusionPipeline` for `from_pretrained()`.

2. **`trust_remote_code=True`**: Automatically injected alongside the custom class, so diffusers allows loading `transformer/transformer_boogu.py` (a custom sub-component code file) during `from_pretrained`.

3. **`remap_input_kwargs` call in `_build_call_kwargs`**: The adapter always produces `{"prompt": ...}` from `_extract_input`. This is now passed through `self._pipeline_utils.remap_input_kwargs()` before the key-acceptance check, allowing pipeline-specific parameter renaming.

**`vllm_omni/diffusion/models/diffusers_adapter/pipeline_utils.py`**

Two additions:

1. **`remap_input_kwargs` hook on `BasePipelineUtils`**: No-op by default. Subclasses override to rename prompt input keys before they reach the `__call__` signature check.

2. **`BooguImagePipelineUtils`**: Remaps `prompt → instruction` and `negative_prompt → negative_instruction`, matching `BooguImagePipeline.__call__`'s actual parameter names. Registered for both `BooguImagePipeline` and `BooguImagePromptTuningPipeline`.

### Serving command

```bash
vllm serve "Boogu/Boogu-Image-0.1-Base" --omni --diffusion-load-format diffusers
```

### Example request

```bash
curl http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Boogu/Boogu-Image-0.1-Base",
    "prompt": "A mountain lake at sunset, photorealistic",
    "n": 1,
    "size": "1024x1024",
    "response_format": "b64_json"
  }'
```

### Model inventory (Step 3)

| Component | Source | Class |
|---|---|---|
| Pipeline | `boogu` package | `BooguImagePipeline` |
| Transformer | Custom (`transformer_boogu.py` stub → `boogu.models.transformers`) | `BooguImageTransformer2DModel` — 3360-dim hidden, 40 layers, 28 heads, 3 weight shards |
| MLLM / text encoder | `transformers` | `Qwen3VLForConditionalGeneration` — 4096-dim, 36 layers, 4 weight shards |
| Scheduler | Custom (`scheduling_flow_match_euler_discrete_time_shifting.py`) | `FlowMatchEulerDiscreteScheduler` with time-shift v1 |
| VAE | `diffusers` | `AutoencoderKL` (standard) |
| Processor | `transformers` | `Qwen3VLProcessor` |

Total model size: ~34.6 GiB on GPU.

---

## Key Validation Principle

Native support is not just "the model loads." A useful native implementation should demonstrate:

- the model can be discovered by vLLM-Omni,
- weights load from the Hugging Face repo or local checkpoint,
- the request API maps correctly into the model pipeline,
- outputs match the upstream Diffusers baseline closely enough,
- online serving works,
- advanced vLLM-Omni features are added only after baseline parity is proven.

