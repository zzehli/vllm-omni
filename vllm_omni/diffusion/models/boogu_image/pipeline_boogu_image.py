# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Native vLLM-Omni pipeline for Boogu-Image-0.1.

Ported from the upstream ``boogu`` package
(``boogu/pipelines/boogu/pipeline_boogu.py``) with the following changes:

- Diffusers ``DiffusionPipeline``/``register_modules`` machinery replaced by a
  plain ``nn.Module`` constructed from ``OmniDiffusionConfig`` (components are
  loaded from the checkpoint subfolders; transformer weights arrive later via
  ``weights_sources`` + ``load_weights``).
- Upstream ``encode_instruction`` is exposed as ``encode_prompt`` (the
  vLLM-Omni convention, also hooked by the prompt-embed cache); only the
  text-to-image path is ported. Instruction rewriting, prompt tuning,
  reference-image (TI2I/I2I) encoding, double guidance, and vision-token
  stripping are not ported.
- ``forward()`` (denoise loop + VAE decode) is intentionally not implemented
  yet; it lands together with the post-process registration.
"""

import json
import os
import warnings
from collections.abc import Iterable
from typing import ClassVar

import torch
import torch.nn.functional as F
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.models.boogu_image.boogu_image_transformer import (
    BooguImageDoubleStreamRotaryPosEmbed,
    BooguImageTransformer2DModel,
)
from vllm_omni.diffusion.models.boogu_image.scheduling_flow_match_euler_discrete_time_shifting import (
    FlowMatchEulerDiscreteScheduler,
)
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

logger = init_logger(__name__)


def get_boogu_image_post_process_func(od_config: OmniDiffusionConfig):
    """Build the post-process callable that converts decoded tensors to images.

    Upstream ``BooguImageProcessor`` only customizes *pre*-processing; the
    ``postprocess`` path is inherited from the stock diffusers
    ``VaeImageProcessor``, so we reuse it directly here.
    """
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])

    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = 2 ** (len(vae_config["block_out_channels"]) - 1) if "block_out_channels" in vae_config else 8

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)

    def post_process_func(images: torch.Tensor):
        return image_processor.postprocess(images)

    return post_process_func


# System prompts matching upstream dataset logic (ported verbatim from
# ``BooguImagePipeline.__init__``).
SYSTEM_PROMPT_4_TI2I_UNIFIED = (
    "Describe the key features of the input image (color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter or modify the image. Generate a new image "
    "that meets the user's requirements while maintaining consistency with the original input where appropriate."
)
SYSTEM_PROMPT_4_T2I_UNIFIED = (
    "You are a helpful assistant that generates high-quality images based on user instructions. "
    "The instructions are as follows."
)


class BooguImagePipeline(nn.Module, ProgressBarMixin, SupportsComponentDiscovery):
    """Boogu-Image text-to-image pipeline (native vLLM-Omni implementation)."""

    supports_request_batch = False

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["mllm"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.od_config = od_config
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

        self._execution_device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)

        # See ``hub_prefetch.py`` for the transformers v5 multi-worker subfolder
        # race; prefetch the whole component set before any from_pretrained.
        boogu_subfolders = ["scheduler", "vae", "mllm", "processor"]
        prefetch_subfolders(model, boogu_subfolders, local_files_only=local_files_only)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )

        mllm = from_pretrained_with_prefetch(
            Qwen3VLForConditionalGeneration.from_pretrained,
            model,
            subfolder="mllm",
            prefetch_list=boogu_subfolders,
            local_files_only=local_files_only,
            torch_dtype=od_config.dtype,
        )
        # Upstream reuses the full VLM as an optional instruction rewriter and
        # encodes with its inner model (no ``lm_head``); the rewriter is not
        # ported, so keep only the inner ``Qwen3VLModel`` as the encoder.
        if hasattr(mllm, "lm_head"):
            mllm = mllm.model
        self.mllm = mllm.to(self._execution_device)

        self.processor = Qwen3VLProcessor.from_pretrained(
            model, subfolder="processor", local_files_only=local_files_only
        )

        self.vae = from_pretrained_with_prefetch(
            AutoencoderKL.from_pretrained,
            model,
            subfolder="vae",
            prefetch_list=boogu_subfolders,
            local_files_only=local_files_only,
        ).to(self._execution_device)

        self.transformer = BooguImageTransformer2DModel(od_config=od_config)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.default_sample_size = 128

        self.SYSTEM_PROMPT_4_T2I = SYSTEM_PROMPT_4_T2I_UNIFIED
        # Upstream uses the TI2I prompt for empty instructions (the default
        # negative prompt "" hits this path).
        self.SYSTEM_PROMPT_DROP = SYSTEM_PROMPT_4_TI2I_UNIFIED

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    # ------------------------------------------------------------------
    # Prompt encoding (upstream ``encode_instruction``, t2i path)
    # ------------------------------------------------------------------

    def _apply_chat_template(self, instruction: str) -> list[dict]:
        """Build the chat messages for one instruction (text-to-image only).

        Mirrors upstream ``_apply_chat_template`` with ``input_pil_images=None``:
        an empty/whitespace instruction selects ``SYSTEM_PROMPT_DROP``.
        """
        if instruction is None or len(instruction.strip()) == 0:
            system_prompt = self.SYSTEM_PROMPT_DROP
        else:
            system_prompt = self.SYSTEM_PROMPT_4_T2I

        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": instruction}]},
        ]

    def _get_instruction_feature_embeds(
        self,
        instruction: str | list[str],
        device: torch.device | None = None,
        max_sequence_length: int = 256,
        truncate_instruction_sequence: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode instructions with the Qwen3VL encoder.

        Returns the last hidden state (or the last-N layers as a list when the
        transformer config asks for more than one) and the attention mask.
        """
        device = device or self._execution_device
        instruction = [instruction] if isinstance(instruction, str) else instruction

        prompts = [self._apply_chat_template(text) for text in instruction]

        vlm_inputs = self.processor.apply_chat_template(
            prompts,
            padding="longest",
            max_length=max_sequence_length,
            truncation=truncate_instruction_sequence,
            padding_side="right",
            return_tensors="pt",
            tokenize=True,
            return_dict=True,
        )
        for k in vlm_inputs.keys():
            if isinstance(vlm_inputs[k], torch.Tensor):
                vlm_inputs[k] = vlm_inputs[k].to(device)

        final_instruction_mask = vlm_inputs["attention_mask"]

        num_instruction_feature_layers = self.transformer.instruction_feature_configs.get(
            "num_instruction_feature_layers", 1
        )

        with torch.no_grad():
            if num_instruction_feature_layers > 1:
                text_encoder_outputs = self.mllm(**vlm_inputs, output_hidden_states=True, return_dict=True)
                instruction_feats = list(text_encoder_outputs.hidden_states)[-num_instruction_feature_layers:]
            else:
                try:
                    instruction_feats = self.mllm(**vlm_inputs, output_hidden_states=False).last_hidden_state
                except Exception as e:
                    text_encoder_outputs = self.mllm(**vlm_inputs, output_hidden_states=True, return_dict=True)
                    instruction_feats = text_encoder_outputs.hidden_states[-1]
                    warnings.warn(f"{type(e).__name__}: {e}", UserWarning, stacklevel=2)

        dtype = self.mllm.dtype if self.mllm is not None else self.transformer.dtype

        if isinstance(instruction_feats, (list, tuple)):
            final_instruction_feats = [feat.to(dtype=dtype, device=device) for feat in instruction_feats]
        else:
            final_instruction_feats = instruction_feats.to(dtype=dtype, device=device)
        final_instruction_mask = final_instruction_mask.to(device=device)

        return final_instruction_feats, final_instruction_mask

    def _reshape_embeds_and_mask(self, embeds, mask, num_images_per_prompt: int):
        """Duplicate embeddings/mask for each generation per prompt (mps-friendly)."""
        if isinstance(embeds, (list, tuple)):
            batch_size, seq_len, _ = embeds[0].shape
            reshaped_embeds = []
            for embed in embeds:
                embed = embed.repeat(1, num_images_per_prompt, 1)
                reshaped_embeds.append(embed.view(batch_size * num_images_per_prompt, seq_len, -1))
        else:
            batch_size, seq_len, _ = embeds.shape
            embeds = embeds.repeat(1, num_images_per_prompt, 1)
            reshaped_embeds = embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        mask = mask.repeat(num_images_per_prompt, 1)
        reshaped_mask = mask.view(batch_size * num_images_per_prompt, -1)

        return batch_size, seq_len, reshaped_embeds, reshaped_mask

    def encode_prompt(
        self,
        prompt: str | list[str],
        do_classifier_free_guidance: bool = True,
        negative_prompt: str | list[str] | None = None,
        num_images_per_prompt: int = 1,
        device: torch.device | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        max_sequence_length: int = 1280,
        truncate_instruction_sequence: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Encode prompt (and negative prompt for CFG) into Qwen3VL hidden states.

        Port of upstream ``encode_instruction`` restricted to text-to-image:
        no reference images, no instruction rewriting, no prompt tuning, no
        double-guidance empty instruction. The default ``max_sequence_length``
        matches the upstream ``__call__`` default (1280), not the upstream
        ``encode_instruction`` default (256).

        Returns:
            ``(prompt_embeds, prompt_attention_mask, negative_prompt_embeds,
            negative_prompt_attention_mask)`` where each embeds tensor has shape
            ``[batch_size * num_images_per_prompt, seq_len, dim]``. The negative
            pair is ``None`` when ``do_classifier_free_guidance`` is off and no
            precomputed negative embeddings were passed.
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_instruction_feature_embeds(
                instruction=prompt,
                device=device,
                max_sequence_length=max_sequence_length,
                truncate_instruction_sequence=truncate_instruction_sequence,
            )

        batch_size, _, prompt_embeds, prompt_attention_mask = self._reshape_embeds_and_mask(
            prompt_embeds, prompt_attention_mask, num_images_per_prompt
        )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt if negative_prompt is not None else ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt` has batch size {len(negative_prompt)}, but `prompt` has"
                    f" batch size {batch_size}. Please make sure that passed `negative_prompt`"
                    " matches the batch size of `prompt`."
                )

            negative_prompt_embeds, negative_prompt_attention_mask = self._get_instruction_feature_embeds(
                instruction=negative_prompt,
                device=device,
                max_sequence_length=max_sequence_length,
                truncate_instruction_sequence=truncate_instruction_sequence,
            )

            _, _, negative_prompt_embeds, negative_prompt_attention_mask = self._reshape_embeds_and_mask(
                negative_prompt_embeds, negative_prompt_attention_mask, num_images_per_prompt
            )

        return (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        )

    # ------------------------------------------------------------------
    # Denoise loop + VAE decode (upstream ``__call__`` / ``processing``, t2i)
    # ------------------------------------------------------------------

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        """Sample initial noise latents (upstream ``prepare_latents``)."""
        height = int(height) // self.vae_scale_factor
        width = int(width) // self.vae_scale_factor
        shape = (batch_size, num_channels_latents, height, width)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        return latents

    def _resolve_output_size(self, height, width):
        """t2i branch of upstream ``_resolve_output_and_original_size``.

        Clamps the working resolution to ``max_input_image_pixels`` (2048**2),
        rounding down to a multiple of ``vae_scale_factor * 2``; the requested
        size is remembered so the decoded image can be resized back.
        """
        img_scale_num = self.vae_scale_factor * 2
        ori_height, ori_width = height, width
        max_pixels = 2048 * 2048
        cur_pixels = height * width
        ratio = min((max_pixels / cur_pixels) ** 0.5, 1.0)
        height = int(height * ratio) // img_scale_num * img_scale_num
        width = int(width * ratio) // img_scale_num * img_scale_num
        return height, width, ori_height, ori_width

    def predict(self, t, latents, instruction_embeds, freqs_cis, instruction_attention_mask):
        """One transformer velocity prediction (upstream ``predict``, t2i)."""
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        return self.transformer(
            latents,
            timestep,
            instruction_embeds,
            freqs_cis,
            instruction_attention_mask,
            ref_image_hidden_states=None,
        )

    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        # Prompt / negative-prompt extraction (mirrors the Ovis pattern; the
        # online API sometimes passes ``{"negative_prompt": None}``).
        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in req.prompts]
        if all(isinstance(p, str) or p.get("negative_prompt") is None for p in req.prompts):
            negative_prompt = None
        else:
            negative_prompt = ["" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in req.prompts]

        sp = req.sampling_params
        device = self._execution_device

        height = sp.height or self.default_sample_size * self.vae_scale_factor
        width = sp.width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = sp.num_inference_steps or 50
        # Upstream default text guidance is 4.0; the engine coerces an unset
        # guidance_scale to 1.0, so only honor a caller-provided value.
        text_guidance_scale = sp.guidance_scale if sp.guidance_scale_provided else 4.0
        num_images_per_prompt = sp.num_outputs_per_prompt if sp.num_outputs_per_prompt > 0 else 1
        generator = sp.generator
        max_sequence_length = sp.max_sequence_length or 1280
        output_type = sp.output_type or "pil"
        cfg_range = (0.0, 1.0)

        do_classifier_free_guidance = text_guidance_scale > 1.0

        batch_size = len(prompt)

        # 1. Encode prompts.
        (
            instruction_embeds,
            instruction_attention_mask,
            negative_instruction_embeds,
            negative_instruction_attention_mask,
        ) = self.encode_prompt(
            prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            max_sequence_length=max_sequence_length,
        )

        # 2. Resolve working / output resolution.
        height, width, ori_height, ori_width = self._resolve_output_size(height, width)

        # 3. Prepare latents.
        dtype = self.vae.dtype
        latent_channels = self.transformer.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            height,
            width,
            instruction_embeds.dtype,
            device,
            generator,
        )

        freqs_cis = BooguImageDoubleStreamRotaryPosEmbed.get_freqs_cis(
            self.transformer.axes_dim_rope,
            self.transformer.axes_lens,
            theta=10000,
        )

        # 4. Timesteps (the ported scheduler consumes ``num_tokens``).
        num_tokens = latents.shape[-2] * latents.shape[-1]
        self.scheduler.set_timesteps(num_inference_steps, device=device, num_tokens=num_tokens)
        timesteps = self.scheduler.timesteps
        num_timesteps = len(timesteps)

        # 5. Denoise loop with (sequential) classifier-free guidance.
        with self.progress_bar(total=num_timesteps) as progress_bar:
            for i, t in enumerate(timesteps):
                model_pred = self.predict(t, latents, instruction_embeds, freqs_cis, instruction_attention_mask)

                in_cfg_range = cfg_range[0] <= i / num_timesteps <= cfg_range[1]
                if do_classifier_free_guidance and in_cfg_range:
                    model_pred_uncond = self.predict(
                        t, latents, negative_instruction_embeds, freqs_cis, negative_instruction_attention_mask
                    )
                    model_pred = model_pred + (text_guidance_scale - 1) * (model_pred - model_pred_uncond)

                latents = self.scheduler.step(model_pred, t, latents, return_dict=False)[0]
                latents = latents.to(dtype=instruction_embeds.dtype)
                progress_bar.update()

        # 6. Decode.
        if output_type == "latent":
            image = latents
        else:
            latents = latents.to(dtype=dtype)
            if self.vae.config.scaling_factor is not None:
                latents = latents / self.vae.config.scaling_factor
            if self.vae.config.shift_factor is not None:
                latents = latents + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            if (ori_height, ori_width) != (height, width):
                image = F.interpolate(image, size=(ori_height, ori_width), mode="bilinear")

        return DiffusionOutput(output=image)
