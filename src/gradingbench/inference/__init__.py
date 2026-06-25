from gradingbench.inference.api import main as run_api_inference
from gradingbench.inference.model_init import call_api, get_llm_inputs, init_open_source_model
from gradingbench.inference.vllm import main as run_vllm_inference

__all__ = [
    "run_api_inference",
    "run_vllm_inference",
    "call_api",
    "get_llm_inputs",
    "init_open_source_model",
]
