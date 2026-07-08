# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""L1 unit tests for the native Boogu-Image pipeline.

Two groups:

1. Constructor tests: all ``from_pretrained`` calls are mocked (Ovis pattern)
   so the pipeline ``__init__`` wiring is exercised without downloading
   weights or building the real transformer.
2. Prompt-encoding tests: a pipeline shell (``object.__new__``) is wired with
   small deterministic fakes so the ported chat templating, processor kwargs,
   CFG handling, and reshape logic can be verified numerically on CPU.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_MODULE = "vllm_omni.diffusion.models.boogu_image.pipeline_boogu_image"

_EMBED_DIM = 8
_SEQ_LEN = 6


# ---------------------------------------------------------------------------
# Constructor tests (mocked from_pretrained)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_dependencies(mocker, monkeypatch):
    """Mock external components so ``__init__`` runs without real weights."""
    # The full VLM wrapper has an ``lm_head``; the pipeline must strip it and
    # keep the inner ``.model`` as the encoder.
    inner_encoder = mocker.MagicMock(name="inner_qwen3vl_model")
    inner_encoder.dtype = torch.float32
    mllm_wrapper = mocker.MagicMock(name="qwen3vl_wrapper")
    mllm_wrapper.model.to.return_value = inner_encoder

    mock_processor = mocker.MagicMock(name="processor")

    mock_vae = mocker.MagicMock(name="vae")
    mock_vae.config.block_out_channels = [128, 256, 512, 512]  # scale factor 8
    mock_vae.to.return_value = mock_vae

    mock_scheduler = mocker.MagicMock(name="scheduler")

    monkeypatch.setattr(
        f"{_MODULE}.FlowMatchEulerDiscreteScheduler.from_pretrained",
        lambda *a, **k: mock_scheduler,
    )
    monkeypatch.setattr(
        f"{_MODULE}.Qwen3VLForConditionalGeneration.from_pretrained",
        lambda *a, **k: mllm_wrapper,
    )
    monkeypatch.setattr(
        f"{_MODULE}.Qwen3VLProcessor.from_pretrained",
        lambda *a, **k: mock_processor,
    )
    monkeypatch.setattr(
        f"{_MODULE}.AutoencoderKL.from_pretrained",
        lambda *a, **k: mock_vae,
    )

    mock_transformer_cls = mocker.MagicMock(name="transformer_cls")
    mock_transformer_instance = mocker.MagicMock(name="transformer")
    mock_transformer_cls.return_value = mock_transformer_instance
    monkeypatch.setattr(f"{_MODULE}.BooguImageTransformer2DModel", mock_transformer_cls)

    # Treat the dummy model id as a local path: skips hub prefetch.
    mocker.patch("os.path.exists", return_value=True)

    return {
        "inner_encoder": inner_encoder,
        "mllm_wrapper": mllm_wrapper,
        "processor": mock_processor,
        "vae": mock_vae,
        "scheduler": mock_scheduler,
        "transformer": mock_transformer_instance,
    }


@pytest.fixture
def boogu_pipeline(mock_dependencies):
    from vllm_omni.diffusion.models.boogu_image.pipeline_boogu_image import (
        BooguImagePipeline,
    )

    od_config = OmniDiffusionConfig(
        model="dummy-boogu",
        tf_model_config=TransformerConfig(params={}),
        dtype=torch.float32,
        num_gpus=1,
    )
    return BooguImagePipeline(od_config=od_config)


def test_boogu_image_pipeline_import():
    from vllm_omni.diffusion.models.boogu_image import BooguImagePipeline

    assert BooguImagePipeline is not None


def test_component_discovery_declarations():
    from vllm_omni.diffusion.models.boogu_image import BooguImagePipeline

    # CPU offload / HSDP discovery must find ``mllm`` (there is no
    # ``text_encoder`` attribute on this pipeline).
    assert BooguImagePipeline._dit_modules == ["transformer"]
    assert BooguImagePipeline._encoder_modules == ["mllm"]
    assert BooguImagePipeline._vae_modules == ["vae"]


def test_constructor_wires_components(boogu_pipeline, mock_dependencies):
    assert boogu_pipeline.scheduler is mock_dependencies["scheduler"]
    assert boogu_pipeline.processor is mock_dependencies["processor"]
    assert boogu_pipeline.vae is mock_dependencies["vae"]
    assert boogu_pipeline.transformer is mock_dependencies["transformer"]
    assert boogu_pipeline.vae_scale_factor == 8
    assert boogu_pipeline.default_sample_size == 128
    assert hasattr(boogu_pipeline, "load_weights")


def test_constructor_strips_mllm_lm_head(boogu_pipeline, mock_dependencies):
    # Upstream encodes with the inner Qwen3VLModel, not the generation wrapper.
    assert boogu_pipeline.mllm is mock_dependencies["inner_encoder"]


def test_constructor_weights_sources(boogu_pipeline):
    (source,) = boogu_pipeline.weights_sources
    assert source.model_or_path == "dummy-boogu"
    assert source.subfolder == "transformer"
    assert source.prefix == "transformer."
    assert source.fall_back_to_pt is True


# ---------------------------------------------------------------------------
# Prompt-encoding tests (deterministic fakes, no constructor)
# ---------------------------------------------------------------------------


class _RecordingProcessor:
    """Fake Qwen3VLProcessor: deterministic token ids derived from the text."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, prompts, **kwargs):
        self.calls.append({"prompts": prompts, "kwargs": kwargs})
        batch = len(prompts)
        input_ids = torch.zeros(batch, _SEQ_LEN, dtype=torch.long)
        for i, messages in enumerate(prompts):
            system_text = messages[0]["content"][0]["text"]
            user_text = messages[1]["content"][0]["text"]
            input_ids[i, 0] = len(system_text) % 997
            input_ids[i, 1] = len(user_text) % 997
            input_ids[i, 2:] = torch.arange(2, _SEQ_LEN) + i
        attention_mask = torch.ones(batch, _SEQ_LEN, dtype=torch.long)
        attention_mask[:, -1] = 0  # fake right-padding
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _FakeMLLM:
    """Fake Qwen3VLModel: hidden states are a deterministic function of ids."""

    dtype = torch.bfloat16

    def __call__(self, input_ids=None, attention_mask=None, output_hidden_states=False, **kwargs):
        base = input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, _EMBED_DIM)
        hidden = base + torch.arange(_EMBED_DIM, dtype=torch.float32)
        return SimpleNamespace(last_hidden_state=hidden)


def _make_encode_pipeline(num_instruction_feature_layers: int = 1):
    from vllm_omni.diffusion.models.boogu_image.pipeline_boogu_image import (
        SYSTEM_PROMPT_4_T2I_UNIFIED,
        SYSTEM_PROMPT_4_TI2I_UNIFIED,
        BooguImagePipeline,
    )

    pipeline = object.__new__(BooguImagePipeline)
    nn.Module.__init__(pipeline)
    pipeline._execution_device = torch.device("cpu")
    pipeline.processor = _RecordingProcessor()
    pipeline.mllm = _FakeMLLM()
    pipeline.transformer = SimpleNamespace(
        instruction_feature_configs={
            "instruction_feat_dim": _EMBED_DIM,
            "num_instruction_feature_layers": num_instruction_feature_layers,
            "reduce_type": "mean",
        },
        dtype=torch.float32,
    )
    pipeline.SYSTEM_PROMPT_4_T2I = SYSTEM_PROMPT_4_T2I_UNIFIED
    pipeline.SYSTEM_PROMPT_DROP = SYSTEM_PROMPT_4_TI2I_UNIFIED
    return pipeline


def test_apply_chat_template_system_prompt_selection():
    pipeline = _make_encode_pipeline()

    messages = pipeline._apply_chat_template("a cat on a mat")
    assert messages[0]["role"] == "system"
    assert messages[0]["content"][0]["text"] == pipeline.SYSTEM_PROMPT_4_T2I
    assert messages[1]["role"] == "user"
    assert messages[1]["content"][0]["text"] == "a cat on a mat"

    # Empty and whitespace-only instructions select the DROP prompt (this is
    # the path the default negative prompt "" takes).
    for empty in ("", "   ", None):
        messages = pipeline._apply_chat_template(empty)
        assert messages[0]["content"][0]["text"] == pipeline.SYSTEM_PROMPT_DROP


def test_processor_called_with_upstream_kwargs():
    pipeline = _make_encode_pipeline()
    pipeline.encode_prompt("a dog", do_classifier_free_guidance=False)

    (call,) = pipeline.processor.calls
    kwargs = call["kwargs"]
    assert kwargs["padding"] == "longest"
    assert kwargs["padding_side"] == "right"
    assert kwargs["truncation"] is False
    assert kwargs["max_length"] == 1280  # upstream __call__ default
    assert kwargs["tokenize"] is True
    assert kwargs["return_dict"] is True
    assert kwargs["return_tensors"] == "pt"


def test_encode_prompt_shapes_and_dtype():
    pipeline = _make_encode_pipeline()
    embeds, mask, neg_embeds, neg_mask = pipeline.encode_prompt(["a dog", "a cat"])

    assert embeds.shape == (2, _SEQ_LEN, _EMBED_DIM)
    assert mask.shape == (2, _SEQ_LEN)
    assert neg_embeds.shape == (2, _SEQ_LEN, _EMBED_DIM)
    assert neg_mask.shape == (2, _SEQ_LEN)
    # Cast to the MLLM dtype, mask passed through from the processor.
    assert embeds.dtype == torch.bfloat16
    assert neg_embeds.dtype == torch.bfloat16
    assert torch.equal(mask[:, -1], torch.zeros(2, dtype=torch.long))


def test_encode_prompt_cfg_negative_default_is_empty_string():
    pipeline = _make_encode_pipeline()
    pipeline.encode_prompt("a dog")

    assert len(pipeline.processor.calls) == 2
    negative_messages = pipeline.processor.calls[1]["prompts"]
    assert len(negative_messages) == 1
    # The default "" negative prompt goes through the DROP system prompt.
    assert negative_messages[0][0]["content"][0]["text"] == pipeline.SYSTEM_PROMPT_DROP
    assert negative_messages[0][1]["content"][0]["text"] == ""


def test_encode_prompt_without_cfg_skips_negative():
    pipeline = _make_encode_pipeline()
    embeds, mask, neg_embeds, neg_mask = pipeline.encode_prompt("a dog", do_classifier_free_guidance=False)

    assert len(pipeline.processor.calls) == 1
    assert embeds.shape == (1, _SEQ_LEN, _EMBED_DIM)
    assert neg_embeds is None
    assert neg_mask is None


def test_encode_prompt_explicit_negative_prompt():
    pipeline = _make_encode_pipeline()
    pipeline.encode_prompt("a dog", negative_prompt="blurry, low quality")

    negative_messages = pipeline.processor.calls[1]["prompts"]
    assert negative_messages[0][0]["content"][0]["text"] == pipeline.SYSTEM_PROMPT_4_T2I
    assert negative_messages[0][1]["content"][0]["text"] == "blurry, low quality"


def test_encode_prompt_negative_batch_mismatch_raises():
    pipeline = _make_encode_pipeline()
    with pytest.raises(ValueError, match="batch size"):
        pipeline.encode_prompt(["a dog", "a cat"], negative_prompt=["only one"])


def test_encode_prompt_num_images_per_prompt_repeats_embeds():
    pipeline = _make_encode_pipeline()
    embeds, mask, neg_embeds, neg_mask = pipeline.encode_prompt("a dog", num_images_per_prompt=3)

    assert embeds.shape == (3, _SEQ_LEN, _EMBED_DIM)
    assert mask.shape == (3, _SEQ_LEN)
    assert torch.equal(embeds[0], embeds[1])
    assert torch.equal(embeds[0], embeds[2])
    assert torch.equal(mask[0], mask[1])
    assert neg_embeds.shape == (3, _SEQ_LEN, _EMBED_DIM)


def test_encode_prompt_precomputed_embeds_bypass_encoder():
    pipeline = _make_encode_pipeline()
    precomputed = torch.randn(1, _SEQ_LEN, _EMBED_DIM)
    precomputed_mask = torch.ones(1, _SEQ_LEN, dtype=torch.long)
    neg_precomputed = torch.randn(1, _SEQ_LEN, _EMBED_DIM)

    embeds, mask, neg_embeds, _ = pipeline.encode_prompt(
        "ignored",
        prompt_embeds=precomputed,
        prompt_attention_mask=precomputed_mask,
        negative_prompt_embeds=neg_precomputed,
    )

    assert len(pipeline.processor.calls) == 0
    assert torch.equal(embeds, precomputed)
    assert torch.equal(neg_embeds, neg_precomputed)


def test_reshape_embeds_and_mask_list_branch():
    # Multi-layer configs return a list of per-layer tensors; the reshape
    # helper must handle both forms.
    pipeline = _make_encode_pipeline()
    layer = torch.arange(2 * _SEQ_LEN * _EMBED_DIM, dtype=torch.float32).view(2, _SEQ_LEN, _EMBED_DIM)
    mask = torch.ones(2, _SEQ_LEN, dtype=torch.long)

    batch_size, seq_len, reshaped, reshaped_mask = pipeline._reshape_embeds_and_mask([layer, layer + 1], mask, 2)

    assert batch_size == 2
    assert seq_len == _SEQ_LEN
    assert isinstance(reshaped, list) and len(reshaped) == 2
    assert reshaped[0].shape == (4, _SEQ_LEN, _EMBED_DIM)
    # repeat(1, n, 1).view(b*n, ...) interleaves per-sample: rows [s0, s0, s1, s1]
    assert torch.equal(reshaped[0][0], reshaped[0][1])
    assert torch.equal(reshaped[0][2], reshaped[0][3])


class _FakeTransformer:
    """Fake denoiser: velocity 0 (identity flow) with the config the loop reads."""

    in_channels = 4
    axes_dim_rope = (8, 4, 4)
    axes_lens = (32, 16, 16)
    dtype = torch.float32
    instruction_feature_configs = {
        "instruction_feat_dim": _EMBED_DIM,
        "num_instruction_feature_layers": 1,
        "reduce_type": "mean",
    }

    def __call__(self, latents, timestep, instruction_embeds, freqs_cis, instruction_attention_mask, **kwargs):
        return torch.zeros_like(latents)


class _FakeScheduler:
    def set_timesteps(self, num_inference_steps, device=None, num_tokens=None):
        self.timesteps = torch.linspace(0, 1, num_inference_steps + 1)[:-1]

    def step(self, model_output, t, latents, return_dict=False):
        return (latents,)


class _FakeDecodeVAE:
    dtype = torch.float32

    def __init__(self):
        self.config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0, block_out_channels=[128, 256, 512, 512])

    def decode(self, latents, return_dict=False):
        batch = latents.shape[0]
        return (torch.zeros(batch, 3, 16, 16),)


def _make_forward_pipeline():
    pipeline = _make_encode_pipeline()
    pipeline.transformer = _FakeTransformer()
    pipeline.scheduler = _FakeScheduler()
    pipeline.vae = _FakeDecodeVAE()
    pipeline.vae_scale_factor = 8
    pipeline.default_sample_size = 128
    return pipeline


def _make_request_batch(prompt, **sampling_overrides):
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    sampling = OmniDiffusionSamplingParams(**sampling_overrides)
    return SimpleNamespace(prompts=[prompt], sampling_params=sampling)


def test_forward_returns_diffusion_output():
    from vllm_omni.diffusion.data import DiffusionOutput

    pipeline = _make_forward_pipeline()
    req = _make_request_batch("a cat", height=64, width=64, num_inference_steps=2)

    out = pipeline.forward(req)

    assert isinstance(out, DiffusionOutput)
    assert isinstance(out.output, torch.Tensor)
    assert out.output.shape[0] == 1
    assert torch.isfinite(out.output).all()
    # CFG default is on (text guidance 4.0), so the encoder ran twice (pos + neg).
    assert len(pipeline.processor.calls) == 2


def test_forward_cfg_off_when_guidance_one():
    pipeline = _make_forward_pipeline()
    req = _make_request_batch("a cat", height=64, width=64, num_inference_steps=2, guidance_scale=1.0)
    # guidance_scale=1.0 is falsy-adjacent but explicitly disables CFG; the
    # request layer would set guidance_scale_provided, so emulate that here.
    req.sampling_params.guidance_scale_provided = True

    pipeline.forward(req)

    # Only the positive prompt is encoded when CFG is off.
    assert len(pipeline.processor.calls) == 1
