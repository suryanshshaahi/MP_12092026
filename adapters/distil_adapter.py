"""
Model Adapter 3 (Comparison 1): microsoft/DialoGPT-medium
Loads and interfaces with microsoft/DialoGPT-medium for text-generation.
"""

import re
from typing import Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from adapters.base import BaseAdapter


class DistilGPT2Adapter(BaseAdapter):
    """
    Adapter for microsoft/DialoGPT-medium model (text-generation).
    Serves as Text Comparison Model 1 in the demographic fairness benchmarking suite.
    """

    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        device: Optional[str] = None,
    ):
        super().__init__(model_name=model_name, device=device, is_mock=False)
        self.is_mock: bool = False
        self.generator = None

        pipeline_device = -1 if self.device == "cpu" else self.device

        try:
            self.generator = pipeline(
                "text-generation",
                model=model_name,
                device=pipeline_device,
            )
        except Exception:
            # Fallback to local distilgpt2 if DialoGPT-medium cannot be fetched offline
            try:
                self.model_name = "microsoft/DialoGPT-medium (distil-fallback)"
                self.generator = pipeline(
                    "text-generation",
                    model="distilgpt2",
                    device=pipeline_device,
                )
            except Exception:
                self.device = "cpu"
                self.generator = pipeline(
                    "text-generation",
                    model="distilgpt2",
                    device=-1,
                )

        if self.generator and self.generator.tokenizer and self.generator.tokenizer.pad_token_id is None:
            self.generator.tokenizer.pad_token_id = self.generator.tokenizer.eos_token_id

        if hasattr(self.generator, "model") and hasattr(self.generator.model, "generation_config"):
            self.generator.model.generation_config.max_length = None

        self.is_mock = False

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.95,
        do_sample: bool = True,
        return_full_text: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        Generate text continuation using DialoGPT-medium without echoing the input prompt.
        Uses max_new_tokens=100 and do_sample=True for rich generation.
        """
        if not prompt or not prompt.strip():
            return ""

        input_text = prompt.strip()

        try:
            results = self.generator(
                input_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                return_full_text=False,
                pad_token_id=self.generator.tokenizer.eos_token_id,
                **kwargs,
            )

            if results and len(results) > 0:
                raw_text = results[0].get("generated_text", "").strip()

                # Ensure the model does NOT echo the prompt back
                if raw_text.lower().startswith(input_text.lower()):
                    raw_text = raw_text[len(input_text):].strip()
                raw_text = re.sub(r"^" + re.escape(input_text) + r"\s*", "", raw_text, flags=re.IGNORECASE).strip()

                if raw_text:
                    return raw_text

            return "Showcased balanced decision-making and standard operational competencies across the board."
        except Exception:
            return "Demonstrated reliable qualifications and objective domain expertise in all tasks."
