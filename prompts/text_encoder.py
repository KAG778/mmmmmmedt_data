import os
# Force offline mode BEFORE importing transformers to prevent any network access
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from typing import Iterable, List, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


class PromptTextEncoder:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = None,
        max_length: int = 256,
        trust_remote_code: bool = True,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.trust_remote_code = trust_remote_code

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model, self._use_hidden_states = self._load_model(model_name)
        self.model.eval()
        self.hidden_size = int(getattr(self.model.config, "hidden_size"))

    def _load_model(self, model_name: str):
        device_str = str(self.device)
        dtype = torch.float16 if device_str.startswith("cuda") else torch.float32
        try:
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=dtype,
                local_files_only=True,
            ).to(self.device)
            return model, False
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=dtype,
                local_files_only=True,
            ).to(self.device)
            return model, True

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (last_hidden_state * mask).sum(dim=1) / denom

    def encode_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 16,
        output_dtype: str = "float16",
    ) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.hidden_size), dtype=np.float16)

        all_embeddings: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch_texts = list(texts[start : start + batch_size])
            encoded = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            with torch.no_grad():
                if self._use_hidden_states:
                    outputs = self.model(**encoded, output_hidden_states=True)
                    last_hidden_state = outputs.hidden_states[-1]
                else:
                    outputs = self.model(**encoded)
                    last_hidden_state = outputs.last_hidden_state

            pooled = self._mean_pool(last_hidden_state, encoded["attention_mask"])
            pooled = pooled.float().cpu().numpy()
            all_embeddings.append(pooled)

        output = np.concatenate(all_embeddings, axis=0)
        if output_dtype == "float16":
            return output.astype(np.float16)
        if output_dtype == "float32":
            return output.astype(np.float32)
        raise ValueError(f"Unsupported output_dtype: {output_dtype}")
