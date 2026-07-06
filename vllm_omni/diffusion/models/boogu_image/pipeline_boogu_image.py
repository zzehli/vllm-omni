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

import os
import warnings
from collections.abc import Iterable
from typing import ClassVar

import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from torch import nn
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.models.boogu_image.boogu_image_transformer import BooguImageTransformer2DModel
from vllm_omni.diffusion.models.boogu_image.scheduling_flow_match_euler_discrete_time_shifting import (
    FlowMatchEulerDiscreteScheduler,
)
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery

logger = init_logger(__name__)

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


class BooguImagePipeline(nn.Module, SupportsComponentDiscovery):
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

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "BooguImagePipeline.forward (denoise loop + VAE decode) is not implemented yet; "
            "it lands together with the post-process registration."
        )
