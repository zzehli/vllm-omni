# Boogu-Image

> Text-to-image online serving (Boogu-Image-0.1-Base)

## Summary

- Vendor: Boogu
- Model: `Boogu/Boogu-Image-0.1-Base`
- Task: Text-to-image generation
- Mode: Online serving with the OpenAI-compatible API
- Maintainer: Community

## When to use this recipe

Use this recipe when you want a known-good starting point for serving
`Boogu/Boogu-Image-0.1-Base` with vLLM-Omni's native pipeline (no
`--diffusion-load-format diffusers`, and the upstream `boogu` package is not
required).

Boogu-Image-0.1 is an Apache-2.0 unified image generation and editing model
family. This recipe covers the Base text-to-image checkpoint, which pairs a
Qwen3-VL multimodal encoder with a Diffusion Transformer (DiT) and a flow-match
Euler scheduler with time-shift. It handles photorealistic generation and
Chinese/English text rendering.

## References

- Upstream model card: <https://huggingface.co/Boogu/Boogu-Image-0.1-Base>
- Project page: <https://boogu.org>
- GitHub: <https://github.com/boogu-project/Boogu-Image>
- Related example: [`examples/online_serving/text_to_image/`](../../examples/online_serving/text_to_image/README.md)

## Hardware Support

This recipe documents tested configurations for CUDA GPU serving. The native
pipeline runs single-GPU; multi-GPU parallelism, CPU offload, and cache
acceleration are not yet supported for this model (see Notes).

## GPU

### 1 x A100/H100 (Single GPU, 40GB+ VRAM)

The model footprint is roughly 34.6 GiB on GPU, so a 40GB+ card is recommended.

#### Command

```bash
vllm serve Boogu/Boogu-Image-0.1-Base --omni --port 8091
```

!!! note
    If you hit Out-of-Memory (OOM) on a smaller card, enable VAE slicing and
    tiling to reduce peak memory: `--vae-use-slicing --vae-use-tiling`.

#### Verification

After the server is ready, test with a simple request:

```bash
curl -X POST http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Boogu/Boogu-Image-0.1-Base",
    "prompt": "A mountain lake at sunset, photorealistic, cinematic lighting",
    "size": "1024x1024",
    "num_inference_steps": 28,
    "guidance_scale": 4.0,
    "seed": 42
  }' | jq -r '.data[0].b64_json' | base64 -d > output.png
```

Or via the chat-completions endpoint (parameters go in `extra_body`):

```bash
curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "A mountain lake at sunset, photorealistic, cinematic lighting"}
    ],
    "extra_body": {
      "height": 1024,
      "width": 1024,
      "num_inference_steps": 28,
      "guidance_scale": 4.0,
      "seed": 42
    }
  }' | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2- | base64 -d > output.png
```

#### Notes

- **Memory usage:** ~34.6 GiB on GPU. Use `--vae-use-slicing --vae-use-tiling`
  to trim peak VRAM if needed.
- **Key flags:**
  - `--omni` — enables vLLM-Omni diffusion serving.
- **Guidance:** Boogu-Image uses `guidance_scale` (mapped to the upstream
  `text_guidance_scale`); the default is `4.0`. Classifier-free guidance is
  active whenever `guidance_scale > 1.0`.
- **Recommended settings:** `num_inference_steps=28`-`50`, `guidance_scale=4.0`.
  The model's maximum native resolution is 2K.
- **Known limitations (not yet supported):** CPU offload
  (`--enable-cpu-offload` / `--enable-layerwise-offload`), Cache-DiT
  (`--cache-backend cache_dit`), and multi-GPU parallelism (TP / SP / CFG /
  HSDP) are planned follow-ups and are not validated for this model yet.
