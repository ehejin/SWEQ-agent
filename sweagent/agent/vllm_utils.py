# Python standard library
from typing import Optional

# Third party imports
from vllm import LLM
from vllm.sampling_params import SamplingParams
from vllm.lora.request import LoRARequest

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


def get_vllm_model(model, max_model_len, enforce_eager=True, num_gpus=1, gpu_memory_utilization=None):
    memory_per_model = 0.9 if gpu_memory_utilization is None else gpu_memory_utilization
    return LLM(
        model=model,
        dtype="float16",
        quantization="bitsandbytes",
        load_format="bitsandbytes", 
        enable_lora=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=memory_per_model,
        tensor_parallel_size=num_gpus,
        enforce_eager=enforce_eager
    )


def generate_response(
    chat,
    vllm_model: Optional[LLM] = None,
    peft_dir: Optional[str] = None,
    tools=None,
    use_function_calling=None,
    **generation_kwargs,
):
    """Generate a response from the assistant model.
    
    Args:
        chat: Either a single message or list of messages for batch processing
        local_model: Optional HuggingFace model
        local_tokenizer: Optional HuggingFace tokenizer
        is_api_model: Whether to use API model
        vllm_model: Optional vLLM model
        **generation_kwargs: Additional generation parameters
        
    Returns:
        str or List[str]: Generated response(s)
    """
    assistant_model_name = generation_kwargs['model']
    if isinstance(chat, str):
        chat = [{"role": "user", "content": chat}]
    elif isinstance(chat[0], str):
        chat = [[{"role": "user", "content": message}] for message in chat]

    if self.use_function_callind and tools:
        tool_descriptions = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                name = func["name"]
                desc = func.get("description", "")
                params = json.dumps(func.get("parameters", {}), indent=2)
                tool_descriptions.append(f"Tool: {name}\nDescription: {desc}\nParameters:\n{params}\n")

        tool_prompt = (
            "You may call the following tools by responding with a function call in this format:\n"
            "`CALL <tool_name>(<args in JSON>)`\n\n"
            + "\n\n".join(tool_descriptions)
        )

        tool_msg = {"role": "system", "content": tool_prompt}
        chat = [tool_msg] + chat

    generation_kwargs.pop("model", None)
    sampling_params = convert_to_sampling_params(generation_kwargs)
    lora_request = LoRARequest("interactive_adapter", 1, peft_dir) if peft_dir else None
    responses = vllm_model.chat(
        messages=chat,
        sampling_params=sampling_params,
        lora_request=lora_request
    )
    
    results = []
    for response_set in responses:
        if response_set.outputs:
            results.append(response_set.outputs[0].text)
            # results.extend([c.text for c in response_set.outputs])
        else:
            results.append("")  # Fallback for empty responses
    return results


def convert_to_sampling_params(generation_kwargs: dict) -> SamplingParams:
    """Convert generation kwargs to vllm SamplingParams."""

    # Valid sampling parameter keys from SamplingParams class
    valid_params = {
        "n",
        "best_of",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "seed",
        "stop",
        "stop_token_ids",
        "bad_words",
        "ignore_eos",
        "max_tokens",
        "min_tokens",
        "logprobs",
        "prompt_logprobs",
        "detokenize",
        "skip_special_tokens",
        "spaces_between_special_tokens",
        "truncate_prompt_tokens",
    }

    # Filter valid params and log unmapped ones
    sampling_kwargs = {}
    for key, value in generation_kwargs.items():
        if key in valid_params:
            sampling_kwargs[key] = value
        else:
            print(
                f"Warning: Parameter '{key}' not found in VLLM-supported sampling parameters"
            )

    # Create SamplingParams object
    return SamplingParams.from_optional(**sampling_kwargs)