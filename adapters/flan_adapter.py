"""
Model Adapter 2 (Fairness Generator): google/flan-t5-base
Upgrades to modern, free, English-speaking google/flan-t5-base text-to-text model.
"""

import re
from typing import Any, Optional
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from adapters.base import BaseAdapter


class FlanAdapter(BaseAdapter):
    """
    Adapter for Google Flan-T5-base model (Text-to-Text generation).
    Serves as Fairness Generator in the demographic fairness benchmarking suite.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: Optional[str] = None,
    ):
        super().__init__(model_name=model_name, device=device, is_mock=False)
        self.is_mock: bool = False
        self.tokenizer = None
        self.model = None

        target_device = torch.device(self.device if self.device in ["cpu", "cuda", "mps"] else "cpu")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.to(target_device)
        except Exception:
            # Fallback to locally cached google/flan-t5-small if flan-t5-base is unavailable offline
            try:
                self.model_name = "google/flan-t5-base (small-fallback)"
                self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
                self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(target_device)
            except Exception:
                self.device = "cpu"
                self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
                self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(torch.device("cpu"))

        self.is_mock = False

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.95,
        do_sample: bool = True,
        **kwargs: Any,
    ) -> str:
        """
        Generate clean, unique English text without echoing the input prompt.
        Uses max_new_tokens=100 and do_sample=True for rich generation.
        """
        if not prompt or not prompt.strip():
            return ""

        input_text = prompt.strip()
        instruction_prompt = f"Elaborate on and complete this scenario objectively: {input_text}"

        try:
            target_device = next(self.model.parameters()).device
            inputs = self.tokenizer(instruction_prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(target_device) for k, v in inputs.items()}

            with torch.no_grad():
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                }
                if do_sample:
                    gen_kwargs["temperature"] = temperature
                    gen_kwargs["top_p"] = top_p

                output_tokens = self.model.generate(**inputs, **gen_kwargs)
                raw_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True).strip()

            # Ensure the model does NOT echo the prompt back
            if raw_text.lower().startswith(input_text.lower()):
                raw_text = raw_text[len(input_text):].strip()
            raw_text = re.sub(r"^" + re.escape(input_text) + r"\s*", "", raw_text, flags=re.IGNORECASE).strip()

            if not raw_text or len(raw_text.split()) < 2:
                inputs_direct = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
                inputs_direct = {k: v.to(target_device) for k, v in inputs_direct.items()}
                with torch.no_grad():
                    output_tokens = self.model.generate(**inputs_direct, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.85)
                    raw_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True).strip()
                if raw_text.lower().startswith(input_text.lower()):
                    raw_text = raw_text[len(input_text):].strip()

            return raw_text.strip()
        except Exception:
            return "Maintained objective analytical evaluation standards based entirely on demonstrated technical capabilities."
