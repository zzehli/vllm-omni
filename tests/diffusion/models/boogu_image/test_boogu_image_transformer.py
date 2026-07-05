def test_boogu_image_transformer_import():
    from vllm_omni.diffusion.models.boogu_image import BooguImageTransformer2DModel

    assert BooguImageTransformer2DModel is not None
