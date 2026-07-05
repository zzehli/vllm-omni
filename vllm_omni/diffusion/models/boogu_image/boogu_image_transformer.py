# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Native vLLM-Omni port of the Boogu-Image transformer.
#
# Ported from the upstream `boogu` package
# (boogu/models/transformers/transformer_boogu.py and friends) with the
# following changes:
#   - Diffusers mixins (ModelMixin/ConfigMixin/PeftAdapterMixin) removed;
#     configuration is read from `od_config.tf_model_config`.
#   - diffusers Attention + custom processors replaced by the vLLM-Omni
#     `Attention` layer (backend/SP/KV-quant handled by the framework).
#   - Q/K/V and FFN projections use vLLM parallel linears for TP support.
#   - Training-time features (prompt tuning, gradient checkpointing) and
#     inference caches (TeaCache/TaylorSeer) are not ported.

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import Timesteps, get_1d_rotary_pos_embed
from einops import repeat
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings in complex form (upstream `use_real=False` path).

    Args:
        x: Query or key tensor of shape [B, S, H, D].
        freqs_cis: Complex frequency tensor of shape [B, S, D // 2].
    """
    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2))
    freqs_cis = freqs_cis.unsqueeze(2)
    x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
    return x_out.type_as(x)


def swiglu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.silu(x.float(), inplace=False).to(x.dtype) * y


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class LuminaRMSNormZero(nn.Module):
    """AdaRMS modulation: projects `temb` into scale/gate terms."""

    def __init__(self, embedding_dim: int, norm_eps: float) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(min(embedding_dim, 1024), 4 * embedding_dim, bias=True)
        self.norm = RMSNorm(embedding_dim, eps=norm_eps)

    def forward(
        self, x: torch.Tensor, emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        scale_msa, gate_msa, scale_mlp, gate_mlp = emb.chunk(4, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None])
        return x, gate_msa, scale_mlp, gate_mlp


class LuminaLayerNormContinuous(nn.Module):
    """Final AdaLN + optional output projection (upstream `norm_out`)."""

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
        out_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear_1 = nn.Linear(conditioning_embedding_dim, embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        self.linear_2 = nn.Linear(embedding_dim, out_dim, bias=bias) if out_dim is not None else None

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        scale = self.linear_1(self.silu(conditioning_embedding).to(x.dtype))
        x = self.norm(x) * (1 + scale)[:, None, :]
        if self.linear_2 is not None:
            x = self.linear_2(x)
        return x


class LuminaFeedForward(nn.Module):
    """SwiGLU feed-forward with tensor-parallel projections."""

    def __init__(
        self,
        dim: int,
        inner_dim: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: float | None = None,
    ) -> None:
        super().__init__()
        if ffn_dim_multiplier is not None:
            inner_dim = int(ffn_dim_multiplier * inner_dim)
        inner_dim = multiple_of * ((inner_dim + multiple_of - 1) // multiple_of)

        self.linear_1 = ColumnParallelLinear(dim, inner_dim, bias=False)  # gate
        self.linear_3 = ColumnParallelLinear(dim, inner_dim, bias=False)  # input
        self.linear_2 = RowParallelLinear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1, _ = self.linear_1(x)
        h2, _ = self.linear_3(x)
        out, _ = self.linear_2(swiglu(h1, h2))
        return out


class Lumina2CombinedTimestepCaptionEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int = 4096,
        instruction_feat_dim: int = 2048,
        frequency_embedding_size: int = 256,
        norm_eps: float = 1e-5,
        timestep_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=frequency_embedding_size,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
            scale=timestep_scale,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=frequency_embedding_size, time_embed_dim=min(hidden_size, 1024)
        )
        self.caption_embedder = nn.Sequential(
            RMSNorm(instruction_feat_dim, eps=norm_eps),
            nn.Linear(instruction_feat_dim, hidden_size, bias=True),
        )

    def forward(
        self,
        timestep: torch.Tensor,
        instruction_hidden_states: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timestep_proj = self.time_proj(timestep).to(dtype=dtype)
        time_embed = self.timestep_embedder(timestep_proj)
        caption_embed = self.caption_embedder(instruction_hidden_states)
        return time_embed, caption_embed


class BooguImageDoubleStreamRotaryPosEmbed(nn.Module):
    """3-axis RoPE producing per-segment frequency tensors.

    Returns separate frequency tensors for instruction tokens, reference-image
    tokens, noise-image tokens, the full joint sequence, and the combined
    (ref + noise) image sequence used by the double-stream blocks.
    """

    def __init__(
        self,
        theta: int,
        axes_dim: tuple[int, int, int],
        axes_lens: tuple[int, int, int] = (300, 512, 512),
        patch_size: int = 2,
    ) -> None:
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.axes_lens = axes_lens
        self.patch_size = patch_size

    @staticmethod
    def get_freqs_cis(
        axes_dim: tuple[int, int, int], axes_lens: tuple[int, int, int], theta: int
    ) -> list[torch.Tensor]:
        freqs_cis = []
        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        for d, e in zip(axes_dim, axes_lens):
            emb = get_1d_rotary_pos_embed(d, e, theta=theta, freqs_dtype=freqs_dtype)
            freqs_cis.append(emb)
        return freqs_cis

    def _get_freqs_cis(self, freqs_cis: list[torch.Tensor], ids: torch.Tensor) -> torch.Tensor:
        device = ids.device
        if ids.device.type == "mps":
            ids = ids.to("cpu")

        result = []
        for i in range(len(self.axes_dim)):
            freqs = freqs_cis[i].to(ids.device)
            index = ids[:, :, i : i + 1].repeat(1, 1, freqs.shape[-1]).to(torch.int64)
            result.append(torch.gather(freqs.unsqueeze(0).repeat(index.shape[0], 1, 1), dim=1, index=index))
        return torch.cat(result, dim=-1).to(device)

    def forward(
        self,
        freqs_cis: list[torch.Tensor],
        attention_mask: torch.Tensor,
        l_effective_ref_img_len: list[list[int]],
        l_effective_img_len: list[int],
        ref_img_sizes: list[list[tuple[int, int]] | None],
        img_sizes: list[tuple[int, int]],
        device: torch.device,
    ):
        batch_size = len(attention_mask)
        p = self.patch_size

        encoder_seq_len = attention_mask.shape[1]
        l_effective_cap_len = attention_mask.sum(dim=1).tolist()

        seq_lengths = [
            cap_len + sum(ref_img_len) + img_len
            for cap_len, ref_img_len, img_len in zip(l_effective_cap_len, l_effective_ref_img_len, l_effective_img_len)
        ]

        max_seq_len = max(seq_lengths)
        max_ref_img_len = max(sum(ref_img_len) for ref_img_len in l_effective_ref_img_len)
        max_img_len = max(l_effective_img_len)

        position_ids = torch.zeros(batch_size, max_seq_len, 3, dtype=torch.int32, device=device)

        for i, (cap_seq_len, seq_len) in enumerate(zip(l_effective_cap_len, seq_lengths)):
            position_ids[i, :cap_seq_len] = repeat(
                torch.arange(cap_seq_len, dtype=torch.int32, device=device), "l -> l 3"
            )

            pe_shift = cap_seq_len
            pe_shift_len = cap_seq_len

            if ref_img_sizes[i] is not None:
                for ref_img_size, ref_img_len in zip(ref_img_sizes[i], l_effective_ref_img_len[i]):
                    H, W = ref_img_size
                    ref_H_tokens, ref_W_tokens = H // p, W // p
                    assert ref_H_tokens * ref_W_tokens == ref_img_len

                    row_ids = repeat(
                        torch.arange(ref_H_tokens, dtype=torch.int32, device=device),
                        "h -> h w",
                        w=ref_W_tokens,
                    ).flatten()
                    col_ids = repeat(
                        torch.arange(ref_W_tokens, dtype=torch.int32, device=device),
                        "w -> h w",
                        h=ref_H_tokens,
                    ).flatten()
                    position_ids[i, pe_shift_len : pe_shift_len + ref_img_len, 0] = pe_shift
                    position_ids[i, pe_shift_len : pe_shift_len + ref_img_len, 1] = row_ids
                    position_ids[i, pe_shift_len : pe_shift_len + ref_img_len, 2] = col_ids

                    pe_shift += max(ref_H_tokens, ref_W_tokens)
                    pe_shift_len += ref_img_len

            H, W = img_sizes[i]
            H_tokens, W_tokens = H // p, W // p
            assert H_tokens * W_tokens == l_effective_img_len[i]

            row_ids = repeat(torch.arange(H_tokens, dtype=torch.int32, device=device), "h -> h w", w=W_tokens).flatten()
            col_ids = repeat(torch.arange(W_tokens, dtype=torch.int32, device=device), "w -> h w", h=H_tokens).flatten()

            assert pe_shift_len + l_effective_img_len[i] == seq_len
            position_ids[i, pe_shift_len:seq_len, 0] = pe_shift
            position_ids[i, pe_shift_len:seq_len, 1] = row_ids
            position_ids[i, pe_shift_len:seq_len, 2] = col_ids

        freqs_cis = self._get_freqs_cis(freqs_cis, position_ids)

        cap_freqs_cis = torch.zeros(
            batch_size, encoder_seq_len, freqs_cis.shape[-1], device=device, dtype=freqs_cis.dtype
        )
        ref_img_freqs_cis = torch.zeros(
            batch_size, max_ref_img_len, freqs_cis.shape[-1], device=device, dtype=freqs_cis.dtype
        )
        img_freqs_cis = torch.zeros(batch_size, max_img_len, freqs_cis.shape[-1], device=device, dtype=freqs_cis.dtype)

        combined_img_seq_lengths = [
            sum(ref_img_len) + img_len for ref_img_len, img_len in zip(l_effective_ref_img_len, l_effective_img_len)
        ]
        max_combined_img_len = max(combined_img_seq_lengths)

        combined_img_freqs_cis = torch.zeros(
            batch_size, max_combined_img_len, freqs_cis.shape[-1], device=device, dtype=freqs_cis.dtype
        )

        for i, (cap_seq_len, ref_img_len, img_len, seq_len) in enumerate(
            zip(l_effective_cap_len, l_effective_ref_img_len, l_effective_img_len, seq_lengths)
        ):
            cap_freqs_cis[i, :cap_seq_len] = freqs_cis[i, :cap_seq_len]
            ref_img_freqs_cis[i, : sum(ref_img_len)] = freqs_cis[i, cap_seq_len : cap_seq_len + sum(ref_img_len)]
            img_freqs_cis[i, :img_len] = freqs_cis[
                i, cap_seq_len + sum(ref_img_len) : cap_seq_len + sum(ref_img_len) + img_len
            ]

            combined_img_freqs_cis[i, : sum(ref_img_len)] = freqs_cis[i, cap_seq_len : cap_seq_len + sum(ref_img_len)]
            combined_img_freqs_cis[i, sum(ref_img_len) : sum(ref_img_len) + img_len] = freqs_cis[
                i, cap_seq_len + sum(ref_img_len) : cap_seq_len + sum(ref_img_len) + img_len
            ]

        return (
            cap_freqs_cis,
            ref_img_freqs_cis,
            img_freqs_cis,
            freqs_cis,
            l_effective_cap_len,
            seq_lengths,
            combined_img_freqs_cis,
            combined_img_seq_lengths,
        )


def _concat_instruction_image_features(
    img_tensors: list[torch.Tensor],
    instruct_tensors: list[torch.Tensor],
    encoder_seq_lengths: list[int],
    seq_lengths: list[int],
) -> list[torch.Tensor]:
    """Pack per-stream tensors into joint sequences (instruction first, then image)."""
    assert len(img_tensors) == len(instruct_tensors)

    batch_size = img_tensors[0].shape[0]
    max_seq_len = max(seq_lengths)

    concatenated_list = []
    for img_tensor, instruct_tensor in zip(img_tensors, instruct_tensors):
        feature_dim = img_tensor.shape[-1]
        concatenated = img_tensor.new_zeros(batch_size, max_seq_len, feature_dim)
        for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
            concatenated[i, :encoder_seq_len] = instruct_tensor[i, :encoder_seq_len]
            concatenated[i, encoder_seq_len:seq_len] = img_tensor[i, : seq_len - encoder_seq_len]
        concatenated_list.append(concatenated)

    return concatenated_list


def _split_instruction_image_features(
    hidden_states: torch.Tensor,
    encoder_seq_lengths: list[int],
    seq_lengths: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack a joint sequence back into (instruction, image) streams."""
    batch_size = hidden_states.shape[0]
    feature_dim = hidden_states.shape[-1]

    max_instruct_len = max(encoder_seq_lengths)
    max_img_len = max(seq_len - encoder_seq_len for seq_len, encoder_seq_len in zip(seq_lengths, encoder_seq_lengths))

    instruct_hidden_states = hidden_states.new_zeros(batch_size, max_instruct_len, feature_dim)
    img_hidden_states = hidden_states.new_zeros(batch_size, max_img_len, feature_dim)

    for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
        img_len = seq_len - encoder_seq_len
        instruct_hidden_states[i, :encoder_seq_len] = hidden_states[i, :encoder_seq_len]
        img_hidden_states[i, :img_len] = hidden_states[i, encoder_seq_len:seq_len]

    return instruct_hidden_states, img_hidden_states


class BooguImageSelfAttention(nn.Module):
    """GQA self-attention with QK RMSNorm and complex RoPE.

    Replaces the upstream diffusers `Attention` + Boogu attention processors;
    the vLLM-Omni `Attention` layer picks the kernel backend and handles
    SP/KV-cache concerns.
    """

    def __init__(self, dim: int, num_attention_heads: int, num_kv_heads: int) -> None:
        super().__init__()
        self.head_dim = dim // num_attention_heads
        kv_dim = self.head_dim * num_kv_heads

        self.to_q = ColumnParallelLinear(dim, dim, bias=False)
        self.to_k = ColumnParallelLinear(dim, kv_dim, bias=False)
        self.to_v = ColumnParallelLinear(dim, kv_dim, bias=False)
        self.norm_q = RMSNorm(self.head_dim, eps=1e-5)
        self.norm_k = RMSNorm(self.head_dim, eps=1e-5)
        self.to_out = RowParallelLinear(dim, dim, bias=False)

        self.num_local_heads = self.to_q.output_size_per_partition // self.head_dim
        self.num_local_kv_heads = self.to_k.output_size_per_partition // self.head_dim

        self.attn = Attention(
            num_heads=self.num_local_heads,
            head_size=self.head_dim,
            softmax_scale=self.head_dim**-0.5,
            causal=False,
            num_kv_heads=self.num_local_kv_heads,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        rotary_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        dtype = hidden_states.dtype

        query, _ = self.to_q(hidden_states)
        key, _ = self.to_k(hidden_states)
        value, _ = self.to_v(hidden_states)

        query = query.unflatten(-1, (self.num_local_heads, self.head_dim))
        key = key.unflatten(-1, (self.num_local_kv_heads, self.head_dim))
        value = value.unflatten(-1, (self.num_local_kv_heads, self.head_dim))

        query = self.norm_q(query)
        key = self.norm_k(key)

        if rotary_emb is not None:
            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)
        query, key = query.to(dtype), key.to(dtype)

        attn_metadata = AttentionMetadata(attn_mask=attention_mask) if attention_mask is not None else None
        attn_output = self.attn(query, key, value, attn_metadata)
        attn_output = attn_output.flatten(2, 3).type_as(hidden_states)

        attn_output, _ = self.to_out(attn_output)
        return attn_output


class BooguImageJointAttention(nn.Module):
    """Joint instruction+image attention for the double-stream blocks.

    The upstream implementation keeps the per-stream Q/K/V and output
    projections on the attention *processor*
    (`BooguImageDoubleStreamSelfAttnProcessor*`); here they are promoted onto
    this module. The final joint output projection corresponds to the upstream
    `img_instruct_attn.to_out[0]`.
    """

    def __init__(self, dim: int, num_attention_heads: int, num_kv_heads: int) -> None:
        super().__init__()
        self.head_dim = dim // num_attention_heads
        kv_dim = self.head_dim * num_kv_heads

        self.img_to_q = ColumnParallelLinear(dim, dim, bias=False)
        self.img_to_k = ColumnParallelLinear(dim, kv_dim, bias=False)
        self.img_to_v = ColumnParallelLinear(dim, kv_dim, bias=False)

        self.instruct_to_q = ColumnParallelLinear(dim, dim, bias=False)
        self.instruct_to_k = ColumnParallelLinear(dim, kv_dim, bias=False)
        self.instruct_to_v = ColumnParallelLinear(dim, kv_dim, bias=False)

        self.norm_q = RMSNorm(self.head_dim, eps=1e-5)
        self.norm_k = RMSNorm(self.head_dim, eps=1e-5)

        # Per-stream output projections (attention output is head-sharded
        # under TP, hence row-parallel).
        self.instruct_out = RowParallelLinear(dim, dim, bias=False)
        self.img_out = RowParallelLinear(dim, dim, bias=False)
        # Final joint projection applied to the merged full-dim sequence.
        self.to_out = ReplicatedLinear(dim, dim, bias=False)

        self.num_local_heads = self.img_to_q.output_size_per_partition // self.head_dim
        self.num_local_kv_heads = self.img_to_k.output_size_per_partition // self.head_dim

        self.attn = Attention(
            num_heads=self.num_local_heads,
            head_size=self.head_dim,
            softmax_scale=self.head_dim**-0.5,
            causal=False,
            num_kv_heads=self.num_local_kv_heads,
        )

    def forward(
        self,
        img_hidden_states: torch.Tensor,
        instruct_hidden_states: torch.Tensor,
        joint_attention_mask: torch.Tensor | None,
        rotary_emb: torch.Tensor | None,
        encoder_seq_lengths: list[int],
        seq_lengths: list[int],
    ) -> torch.Tensor:
        dtype = img_hidden_states.dtype
        batch_size = img_hidden_states.shape[0]

        img_query, _ = self.img_to_q(img_hidden_states)
        img_key, _ = self.img_to_k(img_hidden_states)
        img_value, _ = self.img_to_v(img_hidden_states)

        instruct_query, _ = self.instruct_to_q(instruct_hidden_states)
        instruct_key, _ = self.instruct_to_k(instruct_hidden_states)
        instruct_value, _ = self.instruct_to_v(instruct_hidden_states)

        query, key, value = _concat_instruction_image_features(
            [img_query, img_key, img_value],
            [instruct_query, instruct_key, instruct_value],
            encoder_seq_lengths,
            seq_lengths,
        )

        query = query.view(batch_size, -1, self.num_local_heads, self.head_dim)
        key = key.view(batch_size, -1, self.num_local_kv_heads, self.head_dim)
        value = value.view(batch_size, -1, self.num_local_kv_heads, self.head_dim)

        query = self.norm_q(query)
        key = self.norm_k(key)

        if rotary_emb is not None:
            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)
        query, key = query.to(dtype), key.to(dtype)

        attn_metadata = AttentionMetadata(attn_mask=joint_attention_mask) if joint_attention_mask is not None else None
        attn_output = self.attn(query, key, value, attn_metadata)
        attn_output = attn_output.flatten(2, 3).to(dtype)

        instruct_attn_out, img_attn_out = _split_instruction_image_features(
            attn_output, encoder_seq_lengths, seq_lengths
        )
        instruct_projected, _ = self.instruct_out(instruct_attn_out)
        img_projected, _ = self.img_out(img_attn_out)

        merged = _concat_instruction_image_features(
            [img_projected], [instruct_projected], encoder_seq_lengths, seq_lengths
        )[0]
        merged, _ = self.to_out(merged)
        return merged


class BooguImageTransformerBlock(nn.Module):
    """Single-stream block: GQA attention + SwiGLU FFN with optional AdaRMS modulation."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
        norm_eps: float,
        modulation: bool = True,
    ) -> None:
        super().__init__()
        self.head_dim = dim // num_attention_heads
        self.modulation = modulation

        self.attn = BooguImageSelfAttention(dim, num_attention_heads, num_kv_heads)
        self.feed_forward = LuminaFeedForward(
            dim=dim,
            inner_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        if modulation:
            self.norm1 = LuminaRMSNormZero(embedding_dim=dim, norm_eps=norm_eps)
        else:
            self.norm1 = RMSNorm(dim, eps=norm_eps)

        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        image_rotary_emb: torch.Tensor | None,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.modulation:
            if temb is None:
                raise ValueError("temb must be provided when modulation is enabled")
            norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
            attn_output = self.attn(norm_hidden_states, attention_mask, image_rotary_emb)
            hidden_states = hidden_states + gate_msa.unsqueeze(1).tanh() * self.norm2(attn_output)
            mlp_output = self.feed_forward(self.ffn_norm1(hidden_states) * (1 + scale_mlp.unsqueeze(1)))
            hidden_states = hidden_states + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(mlp_output)
        else:
            norm_hidden_states = self.norm1(hidden_states)
            attn_output = self.attn(norm_hidden_states, attention_mask, image_rotary_emb)
            hidden_states = hidden_states + self.norm2(attn_output)
            mlp_output = self.feed_forward(self.ffn_norm1(hidden_states))
            hidden_states = hidden_states + self.ffn_norm2(mlp_output)

        return hidden_states


class BooguImageNoiseRefinerTransformerBlock(BooguImageTransformerBlock):
    pass


class BooguImageRefImgRefinerTransformerBlock(BooguImageTransformerBlock):
    pass


class BooguImageContextRefinerTransformerBlock(BooguImageTransformerBlock):
    pass


class BooguImageSingleStreamTransformerBlock(BooguImageTransformerBlock):
    pass


class BooguImageDoubleStreamTransformerBlock(nn.Module):
    """Double-stream block: instruction and image tokens are processed in
    parallel streams coupled by a joint attention, plus an extra image
    self-attention."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
        norm_eps: float,
        modulation: bool = True,
    ) -> None:
        super().__init__()
        self.head_dim = dim // num_attention_heads
        self.num_attention_heads = num_attention_heads
        self.modulation = modulation
        self.hidden_size = dim

        self.img_instruct_attn = BooguImageJointAttention(dim, num_attention_heads, num_kv_heads)
        self.img_self_attn = BooguImageSelfAttention(dim, num_attention_heads, num_kv_heads)

        self.img_feed_forward = LuminaFeedForward(
            dim=dim,
            inner_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        if modulation:
            # Image modulation terms: cross-attn, MLP, self-attn.
            self.img_norm1 = LuminaRMSNormZero(embedding_dim=dim, norm_eps=norm_eps)
            self.img_norm2 = LuminaRMSNormZero(embedding_dim=dim, norm_eps=norm_eps)
            self.img_norm3 = LuminaRMSNormZero(embedding_dim=dim, norm_eps=norm_eps)
        else:
            self.img_norm1 = RMSNorm(dim, eps=norm_eps)
            self.img_norm2 = RMSNorm(dim, eps=norm_eps)
            self.img_norm3 = RMSNorm(dim, eps=norm_eps)

        self.img_ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.img_attn_norm = RMSNorm(dim, eps=norm_eps)
        self.img_self_attn_norm = RMSNorm(dim, eps=norm_eps)
        self.img_ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.instruct_feed_forward = LuminaFeedForward(
            dim=dim,
            inner_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        if modulation:
            # Instruction modulation terms: cross-attn, MLP.
            self.instruct_norm1 = LuminaRMSNormZero(embedding_dim=dim, norm_eps=norm_eps)
            self.instruct_norm2 = LuminaRMSNormZero(embedding_dim=dim, norm_eps=norm_eps)
        else:
            self.instruct_norm1 = RMSNorm(dim, eps=norm_eps)
            self.instruct_norm2 = RMSNorm(dim, eps=norm_eps)

        self.instruct_ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.instruct_attn_norm = RMSNorm(dim, eps=norm_eps)
        self.instruct_ffn_norm2 = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        img_hidden_states: torch.Tensor,  # [B, L_img, D] (ref_img + noise_img)
        instruct_hidden_states: torch.Tensor,  # [B, L_instruct, D]
        img_attention_mask: torch.Tensor | None,  # [B, L_img]
        joint_attention_mask: torch.Tensor | None,  # [B, L_total]
        image_rotary_emb: torch.Tensor | None,  # [B, L_img, head_dim // 2]
        rotary_emb: torch.Tensor | None,  # [B, L_total, head_dim // 2]
        temb: torch.Tensor | None = None,
        encoder_seq_lengths: list[int] | None = None,
        seq_lengths: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.modulation and temb is None:
            raise ValueError("temb must be provided when modulation is enabled")

        batch_size = img_hidden_states.shape[0]
        L_instruct = instruct_hidden_states.shape[1]
        L_img = img_hidden_states.shape[1]

        if self.modulation:
            img_norm1_out, img_gate_msa, img_scale_mlp, img_gate_mlp = self.img_norm1(img_hidden_states, temb)
            img_norm2_out, img_shift_mlp, _, _ = self.img_norm2(img_hidden_states, temb)
            img_norm3_out, img_gate_self, _, _ = self.img_norm3(img_hidden_states, temb)

            (
                instruct_norm1_out,
                instruct_gate_msa,
                instruct_scale_mlp,
                instruct_gate_mlp,
            ) = self.instruct_norm1(instruct_hidden_states, temb)
            instruct_norm2_out, instruct_shift_mlp, _, _ = self.instruct_norm2(instruct_hidden_states, temb)

            joint_attn_out = self.img_instruct_attn(
                img_norm1_out,
                instruct_norm1_out,
                joint_attention_mask,
                rotary_emb,
                encoder_seq_lengths,
                seq_lengths,
            )

            # Split the joint output back into instruction/image segments.
            instruct_attn_out = instruct_hidden_states.new_zeros(batch_size, L_instruct, self.hidden_size)
            img_attn_out = img_hidden_states.new_zeros(batch_size, L_img, self.hidden_size)
            for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
                instruct_attn_out[i, :encoder_seq_len] = joint_attn_out[i, :encoder_seq_len]
                img_attn_out[i, : seq_len - encoder_seq_len] = joint_attn_out[i, encoder_seq_len:seq_len]

            img_self_attn_out = self.img_self_attn(img_norm3_out, img_attention_mask, image_rotary_emb)

            img_hidden_states = img_hidden_states + img_gate_msa.unsqueeze(1).tanh() * self.img_attn_norm(img_attn_out)
            img_hidden_states = img_hidden_states + img_gate_self.unsqueeze(1).tanh() * self.img_self_attn_norm(
                img_self_attn_out
            )

            img_mlp_input = (1 + img_scale_mlp.unsqueeze(1)) * img_norm2_out + img_shift_mlp.unsqueeze(1)
            img_mlp_out = self.img_feed_forward(self.img_ffn_norm1(img_mlp_input))
            img_hidden_states = img_hidden_states + img_gate_mlp.unsqueeze(1).tanh() * self.img_ffn_norm2(img_mlp_out)

            instruct_hidden_states = instruct_hidden_states + instruct_gate_msa.unsqueeze(
                1
            ).tanh() * self.instruct_attn_norm(instruct_attn_out)

            instruct_mlp_input = (
                1 + instruct_scale_mlp.unsqueeze(1)
            ) * instruct_norm2_out + instruct_shift_mlp.unsqueeze(1)
            instruct_mlp_out = self.instruct_feed_forward(self.instruct_ffn_norm1(instruct_mlp_input))
            instruct_hidden_states = instruct_hidden_states + instruct_gate_mlp.unsqueeze(
                1
            ).tanh() * self.instruct_ffn_norm2(instruct_mlp_out)

        else:
            img_norm1_out = self.img_norm1(img_hidden_states)
            img_norm3_out = self.img_norm3(img_hidden_states)
            instruct_norm1_out = self.instruct_norm1(instruct_hidden_states)

            joint_attn_out = self.img_instruct_attn(
                img_norm1_out,
                instruct_norm1_out,
                joint_attention_mask,
                rotary_emb,
                encoder_seq_lengths,
                seq_lengths,
            )

            instruct_attn_out = instruct_hidden_states.new_zeros(batch_size, L_instruct, self.hidden_size)
            img_attn_out = img_hidden_states.new_zeros(batch_size, L_img, self.hidden_size)
            for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
                instruct_attn_out[i, :encoder_seq_len] = joint_attn_out[i, :encoder_seq_len]
                img_attn_out[i, : seq_len - encoder_seq_len] = joint_attn_out[i, encoder_seq_len:seq_len]

            img_self_attn_out = self.img_self_attn(img_norm3_out, img_attention_mask, image_rotary_emb)

            img_hidden_states = img_hidden_states + self.img_attn_norm(img_attn_out)
            img_hidden_states = img_hidden_states + self.img_self_attn_norm(img_self_attn_out)
            img_norm2_out = self.img_norm2(img_hidden_states)
            img_mlp_out = self.img_feed_forward(self.img_ffn_norm1(img_norm2_out))
            img_hidden_states = img_hidden_states + self.img_ffn_norm2(img_mlp_out)

            instruct_hidden_states = instruct_hidden_states + self.instruct_attn_norm(instruct_attn_out)
            instruct_norm2_out = self.instruct_norm2(instruct_hidden_states)
            instruct_mlp_out = self.instruct_feed_forward(self.instruct_ffn_norm1(instruct_norm2_out))
            instruct_hidden_states = instruct_hidden_states + self.instruct_ffn_norm2(instruct_mlp_out)

        return img_hidden_states, instruct_hidden_states


def _cal_preprocessed_instruction_feat_dim(instruction_feature_configs: dict) -> int:
    num_instruction_feat_layers = max(instruction_feature_configs.get("num_instruction_feat_layers", 1), 1)
    instruction_feat_dim = instruction_feature_configs.get("instruction_feat_dim", 4096)
    reduce_type = instruction_feature_configs.get("reduce_type", "concat")
    if "cat" in reduce_type.lower():
        return num_instruction_feat_layers * instruction_feat_dim
    elif "mean" in reduce_type.lower():
        return instruction_feat_dim
    else:
        raise ValueError(f"Invalid reduce_type: {reduce_type}")



class BooguImageTransformer2DModel(nn.Module):
    """Boogu-Image transformer with mixed stream topology.

    Early layers use double-stream (dual-stream) processing, then switch to
    single-stream joint processing. Context/noise/reference-image refiner
    blocks run before the main stack.
    """

    # Noise refiner, reference image refiner, and double stream layers are
    # kept out of regional torch.compile for numerical stability (matches the
    # upstream `_repeated_blocks`).
    _repeated_blocks = [
        "BooguImageTransformerBlock",
        "BooguImageContextRefinerTransformerBlock",
        "BooguImageSingleStreamTransformerBlock",
    ]
    _layerwise_offload_blocks_attrs = ["single_stream_layers", "double_stream_layers"]

    def __init__(self, od_config: OmniDiffusionConfig) -> None:
        super().__init__()
        self.od_config = od_config
        cfg = od_config.tf_model_config

        patch_size = cfg.get("patch_size", 2)
        in_channels = cfg.get("in_channels", 16)
        out_channels = cfg.get("out_channels", None)
        hidden_size = cfg.get("hidden_size", 2304)
        num_layers = cfg.get("num_layers", 26)
        num_double_stream_layers = cfg.get("num_double_stream_layers", 2)
        num_refiner_layers = cfg.get("num_refiner_layers", 2)
        num_attention_heads = cfg.get("num_attention_heads", 24)
        num_kv_heads = cfg.get("num_kv_heads", 8)
        multiple_of = cfg.get("multiple_of", 256)
        ffn_dim_multiplier = cfg.get("ffn_dim_multiplier", None)
        norm_eps = cfg.get("norm_eps", 1e-5)
        axes_dim_rope = tuple(cfg.get("axes_dim_rope", (40, 40, 40)))
        axes_lens = tuple(cfg.get("axes_lens", (2048, 1664, 1664)))
        instruction_feature_configs = cfg.get(
            "instruction_feature_configs",
            {"instruction_feat_dim": 1024, "reduce_type": "mean", "num_instruction_feat_layers": 1},
        )
        prompt_tuning_configs = cfg.get("prompt_tuning_configs", {"use_prompt_tuning": False})
        timestep_scale = cfg.get("timestep_scale", 1.0)

        if (hidden_size // num_attention_heads) != sum(axes_dim_rope):
            raise ValueError(
                f"hidden_size // num_attention_heads ({hidden_size // num_attention_heads}) "
                f"must equal sum(axes_dim_rope) ({sum(axes_dim_rope)})"
            )
        if num_double_stream_layers > num_layers:
            raise ValueError(
                f"num_double_stream_layers ({num_double_stream_layers}) cannot be greater than "
                f"num_layers ({num_layers})"
            )
        if prompt_tuning_configs.get("use_prompt_tuning", False):
            raise NotImplementedError(
                "Prompt tuning is a training-time feature and is not supported by the "
                "native vLLM-Omni BooguImageTransformer2DModel."
            )

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_double_stream_layers = num_double_stream_layers
        self.num_single_stream_layers = num_layers - num_double_stream_layers
        self.axes_dim_rope = axes_dim_rope
        self.axes_lens = axes_lens
        self.instruction_feature_configs = instruction_feature_configs
        self.preprocessed_instruction_feat_dim = _cal_preprocessed_instruction_feat_dim(instruction_feature_configs)

        self.rope_embedder = BooguImageDoubleStreamRotaryPosEmbed(
            theta=10000,
            axes_dim=axes_dim_rope,
            axes_lens=axes_lens,
            patch_size=patch_size,
        )

        self.x_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=hidden_size,
        )
        self.ref_image_patch_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=hidden_size,
        )

        self.time_caption_embed = Lumina2CombinedTimestepCaptionEmbedding(
            hidden_size=hidden_size,
            instruction_feat_dim=self.preprocessed_instruction_feat_dim,
            norm_eps=norm_eps,
            timestep_scale=timestep_scale,
        )

        self.noise_refiner = nn.ModuleList(
            [
                BooguImageNoiseRefinerTransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True,
                )
                for _ in range(num_refiner_layers)
            ]
        )

        self.ref_image_refiner = nn.ModuleList(
            [
                BooguImageRefImgRefinerTransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True,
                )
                for _ in range(num_refiner_layers)
            ]
        )

        self.context_refiner = nn.ModuleList(
            [
                BooguImageContextRefinerTransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=False,
                )
                for _ in range(num_refiner_layers)
            ]
        )

        # Mixed architecture: dual-stream first, then single-stream.
        self.double_stream_layers = nn.ModuleList(
            [
                BooguImageDoubleStreamTransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True,
                )
                for _ in range(num_double_stream_layers)
            ]
        )

        self.single_stream_layers = nn.ModuleList(
            [
                BooguImageSingleStreamTransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True,
                )
                for _ in range(self.num_single_stream_layers)
            ]
        )

        self.norm_out = LuminaLayerNormContinuous(
            embedding_dim=hidden_size,
            conditioning_embedding_dim=min(hidden_size, 1024),
            elementwise_affine=False,
            eps=1e-6,
            bias=True,
            out_dim=patch_size * patch_size * self.out_channels,
        )

        # Distinguish multiple reference images (max 5).
        self.image_index_embedding = nn.Parameter(torch.randn(5, hidden_size))

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "BooguImageTransformer2DModel.forward() lands with the pipeline denoising "
            "loop (step 12 of the native support plan)."
        )
