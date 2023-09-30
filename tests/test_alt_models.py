from ai_spider.app import alt_models, adjust_model_for_worker


def test_alt_models():
    alts = alt_models("vicuna-7B-v1.5-GGUF:Q4_K_M")
    assert alts
    assert alts["hf"] == "TheBloke/vicuna-7B-v1.5-GGUF:Q4_K_M"

    alts = alt_models("vicuna-7B-q4")
    assert alts
    assert alts["hf"] == "TheBloke/vicuna-7B-v1.5-GGUF:Q4_K_M"

    alts = alt_models("webgpu/vicuna-7B-q4")
    assert not alts


def test_adjust():
    info = {"web_gpus": [{"name": "3090"}]}
    ok = adjust_model_for_worker("vicuna-7B-v1.5-GGUF:Q4_K_M", info)
    assert ok == "webgpu/vicuna-v1-7b-q4f32_0"
    info = {"nv_gpus": [{"name": "3090"}]}
    ok = adjust_model_for_worker("vicuna-7B-v1.5-GGUF:Q4_K_M", info)
    assert ok == "TheBloke/vicuna-7B-v1.5-GGUF:Q4_K_M"
