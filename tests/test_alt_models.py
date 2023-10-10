from starlette.websockets import WebSocket

from ai_spider.app import alt_models, adjust_model_for_worker, WorkerManager, get_model_size


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


def test_get_sock():
    sock1 = lambda: None
    sock1.info = {}
    info = {"web_gpus": [{"name": "3090"}], "disk_space": 50000, "vram": 8000000000}
    sock2 = lambda: None
    sock2.info = {}
    info2 = {"nv_gpus": [{"name": "3090", "memory": 5000}], "disk_space": 50000, "vram": 8000000000}
    mgr = WorkerManager()
    siz = get_model_size("vicuna-7B-v1.5-GGUF:Q4_K_M")
    assert siz < 5
    mgr.register_js(sock1, info)
    mgr.register_js(sock2, info2)
    with mgr.get_socket_for_inference(siz, "cli", {}) as ws:
        assert ws == sock2
    with mgr.get_socket_for_inference(siz, "web", {}) as ws:
        assert ws == sock1
    with mgr.get_socket_for_inference(siz, "any", {}) as ws:
        adj = adjust_model_for_worker("vicuna-7B-v1.5-GGUF:Q4_K_M", ws.info)
        assert "/" in adj
