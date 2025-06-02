 # vllm_utils.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7"
import json
import threading
import re
from typing import Optional, List, Dict, Union

from vllm import LLM
from vllm.sampling_params import SamplingParams
from vllm.lora.request import LoRARequest


_vllm_init_lock = threading.Lock()
_vllm_singleton: Optional[LLM] = None



def get_vllm_model(
    model: str,
    max_model_len: int,
    enforce_eager: bool = True,
    num_gpus: int = 2,
    gpu_memory_utilization: Optional[float] = None,
) -> LLM:
    """
    Lazily instantiate and return a single shared vLLM instance.
    We use a global lock so that only one thread can run LLM(...) at a time—
    that way, vLLM’s internal 'tensor model parallel group' assertion never races.
    """
    global _vllm_singleton

    # Acquire lock before checking/creating
    with _vllm_init_lock:
        if _vllm_singleton is None:
            memory_per_model = 0.9 if gpu_memory_utilization is None else gpu_memory_utilization
            try:
                _vllm_singleton = LLM(
                    model=model,
                    dtype="float16",
                    quantization="bitsandbytes",
                    load_format="bitsandbytes",
                    enable_lora=True,
                    max_seq_len_to_capture=max_model_len,
                    gpu_memory_utilization=memory_per_model,
                    tensor_parallel_size=num_gpus,
                    enforce_eager=enforce_eager,
                    hf_overrides={
                        "rope_scaling": {
                            "rope_type": "yarn",
                            "factor": 4.0,
                            "original_max_position_embeddings": 32768
                        }
                    },
                )
            except AssertionError as e:
                msg = str(e)
                # If vLLM complains that the tensor‐parallel group is already initialized,
                # just return whatever got set (it might still be None if this is really
                # the very first call and something else pre‐initialized, but nothing
                # better can be done).
                if "tensor model parallel group is already initialized" in msg:
                    return _vllm_singleton  # could be None if no one set it yet, or the already‐created LLM
                else:
                    raise
    return _vllm_singleton


def convert_to_sampling_params(generation_kwargs: dict) -> SamplingParams:
    """
    Convert a dict of generation kwargs into a vLLM SamplingParams object.
    Must include at least "max_tokens" as a positive integer.
    """
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
        "max_tokens",      # ← REQUIRED for vLLM
        "min_tokens",
        "logprobs",
        "prompt_logprobs",
        "detokenize",
        "skip_special_tokens",
        "spaces_between_special_tokens",
        "truncate_prompt_tokens",
    }

    sampling_kwargs = {}
    for key, value in generation_kwargs.items():
        if key in valid_params:
            sampling_kwargs[key] = value
        else:
            # Warn about unsupported params
            print(f"Warning: parameter '{key}' is not supported by vLLM SamplingParams")

    if "max_tokens" not in sampling_kwargs or sampling_kwargs["max_tokens"] is None:
        raise ValueError("vLLM requires a positive 'max_tokens' in generation_kwargs")

    return SamplingParams.from_optional(**sampling_kwargs)


def generate_response(
    chat: Union[str, List[Dict[str, str]]],
    vllm_model: Optional[LLM] = None,
    peft_dir: Optional[str] = None,
    tools: Optional[List[Dict]] = None,
    use_function_calling: Optional[bool] = None,
    **generation_kwargs,
) -> List[Dict]:
    """
    Generate responses via vLLM, then parse them into OpenAI‐style function_call dicts.

    Returns a list of dicts like:
      {
        "role": "assistant",
        "content": "<text or empty string>",
        "function_call": {"name": "<tool_name>", "arguments": "<JSONString>"}  # optional
      }
    """
    # 1) Build batched list of messages for vLLM.chat
    if isinstance(chat, str):
        batched_chat: List[List[Dict[str, str]]] = [[{"role": "user", "content": chat}]]
    elif isinstance(chat, list) and chat and isinstance(chat[0], dict):
        batched_chat = [chat]  # type: ignore
    elif isinstance(chat, list) and chat and isinstance(chat[0], list):
        batched_chat = chat  # type: ignore
    else:
        raise ValueError("Invalid `chat` format. Must be str, List[dict], or List[List[dict]].")

    #print("BATCHED CHAT is: ", batched_chat)
    # 2) Prepend system message if function-calling is enabled
    if use_function_calling and tools:
        tool_descriptions = []
        for tool in tools:
            if isinstance(tool, dict) and tool.get("type") == "function":
                func = tool.get("function", {})
                name = func.get("name", "")
                desc = func.get("description", "")
                params = json.dumps(func.get("parameters", {}), indent=2)
                tool_descriptions.append(
                    f"Tool: {name}\nDescription: {desc}\nParameters:\n{params}\n"
                )

        if tool_descriptions:
            tool_prompt = (
                "You may call exactly one of the following tools by responding with either:\n\n"
                "  1) `CALL <tool_name>(<arguments in JSON>)`\n\n"
                "  2) A JSON object exactly of the form:\n"
                "     {\"name\":\"<tool_name>\",\"arguments\":{ … }}\n\n"
                "No other output is allowed."
                "\n\n"
                + "\n\n".join(tool_descriptions)
            )
            sys_msg = {"role": "system", "content": tool_prompt}
            batched_chat = [[sys_msg] + convo for convo in batched_chat]

    # 3) Remove any accidental "model" key
    generation_kwargs.pop("model", None)

    # 4) Convert kwargs to SamplingParams
    sampling_params = convert_to_sampling_params(generation_kwargs)

    # 5) Build LoRARequest if needed
    lora_request = LoRARequest("interactive_adapter", 1, peft_dir) if peft_dir else None

    # 6) Call vLLM.chat
    responses = vllm_model.chat(
        messages=batched_chat,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )

    #print('RAW RESPONSE: ', responses)

    # 7) Parse each ResponseSet into one function_call dict if present
    out_list: List[Dict] = []
    # Pattern to match "CALL toolName({...})"
    call_pattern = re.compile(r"^CALL\s+([a-zA-Z0-9_]+)\((\{.*\})\)\s*$", re.DOTALL)

    for response_set in responses:
        if response_set.outputs:
            raw_text = response_set.outputs[0].text.strip()
        else:
            raw_text = ""

        # If triple backticks wrap JSON, strip them
        if raw_text.startswith("```"):
            first_newline = raw_text.find("\n")
            if first_newline != -1:
                raw_text = raw_text[first_newline + 1 :]
            if raw_text.endswith("```"):
                raw_text = raw_text[: -3].rstrip()

        # Try matching "CALL tool({...})"
        match = call_pattern.match(raw_text)
        if match:
            tool_name = match.group(1)
            args_json = match.group(2)
        else:
            # Try parsing raw_text as JSON {"name":..., "arguments":{...}}
            try:
                parsed = json.loads(raw_text)
                if (
                    isinstance(parsed, dict)
                    and "name" in parsed
                    and "arguments" in parsed
                    and isinstance(parsed["arguments"], dict)
                ):
                    tool_name = parsed["name"]
                    args_json = json.dumps(parsed["arguments"])
                else:
                    raise ValueError
            except Exception:
                # No function call found; return plain content
                out_list.append({"role": "assistant", "content": raw_text})
                continue

        func_call = {"name": tool_name, "arguments": args_json}
        out_list.append({"role": "assistant", "content": None, "function_call": func_call})

    return out_list


'''from typing import Optional

import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Third party imports
from vllm import LLM
from vllm.sampling_params import SamplingParams
from vllm.lora.request import LoRARequest
from vllm.forward_context import set_forward_context, get_forward_context


# Load environment variables
from dotenv import load_dotenv
load_dotenv()

_vllm_singleton: LLM | None = None
hub_id = "hf://Qwen/Qwen2.5-Coder-7B-Instruct"

def get_vllm_model(model, max_model_len, enforce_eager=True, num_gpus=1, gpu_memory_utilization=None):
    global _vllm_singleton
    print("[VLLM_UTILS] get_vllm_model called; singleton is", "set" if _vllm_singleton else "None")
    if _vllm_singleton is None:
        print("[VLLM_UTILS] Instantiating new vLLM engine")
        memory_per_model = 0.9 if gpu_memory_utilization is None else gpu_memory_utilization
        _vllm_singleton =  LLM(
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
    else:
        print("[VLLM_UTILS] Reusing existing vLLM engine")
    return _vllm_singleton

def get_vllm_model(model, max_model_len, enforce_eager=True, num_gpus=1):
    global _vllm_singleton
    if _vllm_singleton is None:
        _vllm_singleton = LLM(
            model=model,
            dtype="float16",       # cast weights to fp16 at runtime
            quantization=None,     # disable bitsandbytes quant
            load_format="safetensors",      # load PyTorch weights (pt) rather than BnB
            enable_lora=False,     # turn off Triton/LORA entirely
            max_model_len=max_model_len,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=num_gpus,
            pipeline_parallel_size=1,
            enforce_eager=enforce_eager,
        )
    return _vllm_singleton


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
        vllm_model: Optional vLLM model
        peft_dir: Optional LoRA adapter directory
        tools: Either a ToolHandler instance or a list of dicts describing functions
        use_function_calling: Whether to prepend a function‐calling prompt
        **generation_kwargs: Additional generation parameters (must include 'model')
        
    Returns:
        List[str]: Generated response(s)
    """
    assistant_model_name = generation_kwargs["model"]

    # Normalize chat into list-of-messages format
    if isinstance(chat, str):
        chat = [{"role": "user", "content": chat}]
    elif isinstance(chat[0], str):
        chat = [[{"role": "user", "content": message}] for message in chat]

    # If we want vLLM to know about functions, extract the underlying list of dicts
    if use_function_calling and tools:
        # If `tools` is a ToolHandler, grab its `.tools` attribute; otherwise assume it's already a list of dicts
        tool_list = getattr(tools, "tools", tools)

        tool_descriptions = []
        for tool in tool_list:
            # skip anything that isn't a dict
            if not isinstance(tool, dict):
                continue
            if tool.get("type") == "function":
                func = tool["function"]
                name = func["name"]
                desc = func.get("description", "")
                params = json.dumps(func.get("parameters", {}), indent=2)
                tool_descriptions.append(
                    f"Tool: {name}\nDescription: {desc}\nParameters:\n{params}\n"
                )

        if tool_descriptions:
            tool_prompt = (
                "You may call the following tools by responding with a function call in this format:\n"
                "`CALL <tool_name>(<args in JSON>)`\n\n"
                + "\n\n".join(tool_descriptions)
            )
            tool_msg = {"role": "system", "content": tool_prompt}
            chat = [tool_msg] + chat

    # Remove 'model' before converting to SamplingParams
    generation_kwargs.pop("model", None)
    sampling_params = convert_to_sampling_params(generation_kwargs)
    lora_request = LoRARequest("interactive_adapter", 1, peft_dir) if peft_dir else None

    responses = vllm_model.chat(
        messages=chat,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )


    results = []
    for response_set in responses:
        if response_set.outputs:
            results.append(response_set.outputs[0].text)
        else:
            results.append("")  # Fallback for empty responses

    return results


def generate_response_OG(
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

    if use_function_calling and tools:
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
    return SamplingParams.from_optional(**sampling_kwargs)'''