"""Local HuggingFace inference wrapper, API-compatible with AzureInferenceEngine.

Returns a list-of-strings completion so existing run_* scripts can use it
interchangeably with Azure / Together.

Supports any HF causal-LM (Llama-3 / Llama-3.1 / Qwen / Mistral / etc).
For Llama-3-8B-Instruct on a single A6000 / A100-40GB use defaults.
For Llama-3-70B-Instruct add `load_in_4bit=True` and put on 1× A100-80GB,
or set `device_map="auto"` across 2× A100-80GB at fp16.
"""
from __future__ import annotations
import os, time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalHFEngine:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        max_tokens: int = 8,
        temperature: float = 0.0,
        top_p: float = 1.0,
        dtype: torch.dtype = torch.float16,
        device_map: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        verbose: bool = False,
        seed: int = 42,
    ):
        self.model_name  = model_name
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.top_p       = top_p
        self.verbose     = verbose

        load_kwargs = {"torch_dtype": dtype, "device_map": device_map}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs.pop("torch_dtype", None)
        elif load_in_8bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs.pop("torch_dtype", None)

        if verbose: print(f"[LocalHFEngine] loading {model_name} ...")
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token_id = self.tok.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()
        if verbose: print(f"[LocalHFEngine] loaded in {time.time() - t0:.1f}s")

        torch.manual_seed(seed)

    def _chat_prompt(self, user_prompt: str) -> torch.Tensor:
        msgs = [{"role": "user", "content": user_prompt}]
        ids = self.tok.apply_chat_template(
            msgs, return_tensors="pt", add_generation_prompt=True,
        )
        return ids.to(self.model.device)

    @torch.no_grad()
    def __call__(self, prompt: str, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None, **kwargs) -> list[str]:
        ids = self._chat_prompt(prompt)
        mt = max_tokens if max_tokens is not None else self.max_tokens
        T  = temperature if temperature is not None else self.temperature

        out = self.model.generate(
            ids,
            max_new_tokens=mt,
            do_sample=(T > 0),
            temperature=max(T, 1e-6),
            top_p=self.top_p,
            pad_token_id=self.tok.pad_token_id,
        )
        new_tokens = out[0, ids.shape[1] :]
        text = self.tok.decode(new_tokens, skip_special_tokens=True).strip()
        return [text]


# Tiny CLI smoke test
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--4bit", action="store_true", dest="four_bit")
    ap.add_argument("--prompt",
                    default="Output one word: MET, NOT_MET, or UNCLEAR.\n"
                            "CRITERION: Exclude if patient has diabetes.\n"
                            "PATIENT: Patient has type 2 diabetes, A1c 8.4.\n")
    args = ap.parse_args()

    eng = LocalHFEngine(model_name=args.model, load_in_4bit=args.four_bit,
                        verbose=True, max_tokens=10)
    print("OUTPUT:", repr(eng(args.prompt)[0]))
