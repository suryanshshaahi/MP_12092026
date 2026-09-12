"""
Base Model Adapter Module
Provides the abstract base class for all language model adapters in the
Demographic Fairness & Responsible AI Evaluation Suite.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import torch


class BaseAdapter(ABC):
    """
    Abstract Base Class for Large Language Model (LLM) Adapters.
    Defines common initialization logic, device detection, and abstract generation contract.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        is_mock: bool = False,
    ) -> None:
        """
        Initialize the adapter base.

        Args:
            model_name: Hugging Face model identifier or local directory path.
            device: Computing device ('cpu', 'cuda', 'mps', or None for auto-detection).
            is_mock: Boolean indicator of whether fallback mock generation is in effect.
        """
        self.model_name: str = model_name
        self.is_mock: bool = is_mock

        if device is not None:
            self.device: str = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 60, **kwargs: Any) -> str:
        """
        Generate text output for a provided input prompt.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum number of new tokens to generate.
            **kwargs: Model-specific generation hyperparameters.

        Returns:
            str: Generated text continuation.
        """
        raise NotImplementedError("Subclasses must implement generate()")

    def get_info(self) -> dict:
        """
        Retrieve adapter status and metadata.
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "is_mock": self.is_mock,
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(model='{self.model_name}', device='{self.device}', is_mock={self.is_mock})>"
