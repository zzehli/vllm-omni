import pytest


@pytest.fixture
def make_boogu_image_pipeline():
    def _make():
        from vllm_omni.diffusion.models.boogu_image.pipeline_boogu_image import (
            BooguImagePipeline,
        )

        pipeline = object.__new__(BooguImagePipeline)
        return pipeline

    return _make


def test_boogu_image_pipeline_import():
    from vllm_omni.diffusion.models.boogu_image import BooguImagePipeline

    assert BooguImagePipeline is not None


def test_boogu_image_pipeline_instantiates(make_boogu_image_pipeline):
    from vllm_omni.diffusion.models.boogu_image import BooguImagePipeline

    pipeline = make_boogu_image_pipeline()
    assert isinstance(pipeline, BooguImagePipeline)
