import os

# Python 3.12 + vLLM TP workers: force setuptools bundled distutils
os.environ["SETUPTOOLS_USE_DISTUTILS"] = "local"
import setuptools  # noqa: F401

from vllm import LLM, EngineArgs, SamplingParams
from transformers import AutoProcessor
from dataclasses import asdict
import requests

from .utils import encode_image


def call_api(image_path, prompt, model_name):
    """Call a hosted vision-language API (OpenAI-compatible). Set PIGAI_API_APP_ID / PIGAI_API_APP_KEY."""
    from paths import get_api_credentials, get_api_endpoint

    app_id, app_key = get_api_credentials(model_name)
    if not app_id or not app_key:
        raise RuntimeError(
            "API credentials not configured. Set PIGAI_API_APP_ID and PIGAI_API_APP_KEY, "
            "or use main_test_vllm.py for local open-source inference."
        )

    base64_image = encode_image(image_path)
    data = {
        "model": f"{model_name}",
        "messages": [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpg;base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
    }

    headers = {
        "Authorization": f"Bearer {app_id}:{app_key}",
        "Content-Type": "application/json",
    }
    endpoint = get_api_endpoint()
    try:
        response = requests.post(endpoint, headers=headers, json=data, timeout=600)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        print(f"API error HTTP {response.status_code}: {response.text[:200]}")
        return None
    except Exception as e:
        print(f"API request failed: {e}")
        return None


def init_open_source_model(model_name, model_path, tensor_parallel_size):
    """初始化开源 MLLM（vLLM 后端）。"""
    if model_name.startswith("deepseek"):
        engine_args = EngineArgs(
            model=model_path,
            max_model_len=4096,
            max_num_seqs=2,
            hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
            dtype="bfloat16",
            trust_remote_code=True,
            enforce_eager=True,
            gpu_memory_utilization=0.6,
            tensor_parallel_size=tensor_parallel_size,
            limit_mm_per_prompt={"image": 10},
        )
        llm = LLM(**asdict(engine_args))
    elif model_name.startswith(("gemma", "Qwen")):
        llm = LLM(
            model=model_path,
            limit_mm_per_prompt={"image": 10},
            gpu_memory_utilization=0.6,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=32768,
            max_num_seqs=2,
        )
    else:
        llm = LLM(
            model=model_path,
            max_model_len=32768,
            gpu_memory_utilization=0.6,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 10},
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=True,
        )

    if model_name.startswith("deepseek"):
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=4096,
            skip_special_tokens=False,
            repetition_penalty=1.1,
            stop_token_ids=[],
        )
    elif model_name.startswith("gemma"):
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.1,
            max_tokens=8192,
            skip_special_tokens=False,
            stop_token_ids=[],
        )
    elif model_name.startswith("Qwen"):
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.1,
            max_tokens=8192,
            skip_special_tokens=False,
            stop_token_ids=[],
        )
    else:
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.1,
            max_tokens=8192,
            skip_special_tokens=False,
            stop_token_ids=[],
        )

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return llm, sampling_params, processor


def get_llm_inputs(model_name, prompt, image, image_path, processor=None):
    if model_name.startswith("deepseek"):
        full_prompt = f"<|User|>: image_1:<image>\n{prompt}\n<|Assistant|>:"
        if len(full_prompt) > 2500:
            return None
        return {"prompt": full_prompt, "multi_modal_data": {"image": [image]}}

    if model_name.startswith("Qwen"):
        from qwen_vl_utils import process_vision_info

        content = [
            {
                "type": "image",
                "image": image_path,
                "min_pixels": 224 * 224,
                "max_pixels": 2200 * 28 * 28,
            },
            {"type": "text", "text": prompt},
        ]
        messages = [{"role": "system", "content": ""}, {"role": "user", "content": content}]
        image_inputs, _ = process_vision_info(messages)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"prompt": text, "multi_modal_data": {"image": image_inputs}}

    if model_name.startswith("InternVL"):
        full_prompt = f"<image>\n{prompt}"
        messages = [{"role": "system", "content": ""}, {"role": "user", "content": full_prompt}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"prompt": text, "multi_modal_data": {"image": image}}

    content = [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]
    messages = [{"role": "system", "content": ""}, {"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"prompt": text, "multi_modal_data": {"image": image}}
