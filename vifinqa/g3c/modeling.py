"""Lazy GPU backends for the pinned Qwen3 embedding and reranker models."""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import math
import platform
from typing import Protocol

import numpy as np

from ..utils.viet_text import tokens


class EmbeddingBackend(Protocol):
    dimension: int

    def encode(
        self, texts: list[str], *, is_query: bool = False,
        instruction: str = "",
    ) -> np.ndarray: ...

    def close(self) -> None: ...


class RerankerBackend(Protocol):
    def score(
        self, pairs: list[tuple[str, str]], *, instruction: str,
        max_length: int | None = None,
    ) -> list[float]: ...

    def close(self) -> None: ...


class FakeEmbeddingBackend:
    """Deterministic CPU backend for contracts and smoke tests only."""

    def __init__(self, dimension: int = 128):
        self.dimension = int(dimension)

    def encode(
        self, texts: list[str], *, is_query: bool = False,
        instruction: str = "",
    ) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            body = f"{instruction}\n{text}" if is_query else text
            for token in tokens(body):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] & 1 else -1.0
                vectors[row, index] += sign
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors /= np.maximum(norms, 1e-12)
        return vectors

    def close(self) -> None:
        return None


class FakeRerankerBackend:
    """Token-overlap smoke backend; never valid as scientific evidence."""

    def score(
        self, pairs: list[tuple[str, str]], *, instruction: str,
        max_length: int | None = None,
    ) -> list[float]:
        output = []
        for query, document in pairs:
            q = set(tokens(query))
            d = set(tokens(document))
            overlap = len(q & d) / max(1, len(q | d))
            output.append(float(overlap))
        return output

    def close(self) -> None:
        return None


class QwenEmbeddingBackend:
    """Official last-token pooling with L2-normalized Qwen3 embeddings."""

    def __init__(self, model_config: dict, runtime: dict):
        _require_cuda_runtime()
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.dimension = int(model_config["embedding_dimension"])
        self.batch_size = int(runtime["embedding_batch_size"])
        self.max_query_length = int(runtime["query_max_length"])
        self.max_table_length = int(runtime["table_max_length"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config["model_id"],
            revision=model_config["tokenizer_revision"],
            trust_remote_code=False,
            padding_side="left",
        )
        self.model = AutoModel.from_pretrained(
            model_config["model_id"],
            revision=model_config["revision"],
            trust_remote_code=False,
            torch_dtype=torch.float16,
            attn_implementation=runtime["attention_implementation"],
            low_cpu_mem_usage=True,
        ).to("cuda").eval()

    def encode(
        self, texts: list[str], *, is_query: bool = False,
        instruction: str = "",
    ) -> np.ndarray:
        torch = self._torch
        if is_query:
            texts = [
                f"Instruct: {instruction}\nQuery: {text}" for text in texts
            ]
        max_length = (
            self.max_query_length if is_query else self.max_table_length
        )
        output: list[np.ndarray] = []
        with torch.inference_mode():
            starts = range(0, len(texts), self.batch_size)
            for start in _batch_progress(
                starts, "Qwen embedding queries" if is_query
                else "Qwen embedding passages"
            ):
                batch = texts[start:start + self.batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to("cuda")
                result = self.model(**encoded)
                vectors = _last_token_pool(
                    result.last_hidden_state, encoded["attention_mask"]
                )
                vectors = torch.nn.functional.normalize(
                    vectors.float(), p=2, dim=1
                )
                output.append(vectors.cpu().numpy().astype(np.float16))
        if not output:
            return np.empty((0, self.dimension), dtype=np.float16)
        result = np.concatenate(output, axis=0)
        if result.shape[1] != self.dimension:
            raise RuntimeError(
                f"embedding dimension {result.shape[1]} != {self.dimension}"
            )
        return result

    def close(self) -> None:
        model = getattr(self, "model", None)
        tokenizer = getattr(self, "tokenizer", None)
        if model is not None:
            del self.model
        if tokenizer is not None:
            del self.tokenizer
        gc.collect()
        self._torch.cuda.empty_cache()


class QwenRerankerBackend:
    """Qwen3 CausalLM yes/no relevance probability from final-token logits."""

    _PREFIX = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query "
        "and the Instruct provided. Note that the answer can only be "
        "\"yes\" or \"no\".<|im_end|>\n"
        "<|im_start|>user\n"
    )
    _SUFFIX = (
        "<|im_end|>\n<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )

    def __init__(self, model_config: dict, runtime: dict):
        _require_cuda_runtime()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.batch_size = int(runtime["reranker_batch_size"])
        self.max_length = int(runtime["reranker_max_length"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config["model_id"],
            revision=model_config["tokenizer_revision"],
            trust_remote_code=False,
            padding_side="left",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config["model_id"],
            revision=model_config["revision"],
            trust_remote_code=False,
            torch_dtype=torch.float16,
            attn_implementation=runtime["attention_implementation"],
            low_cpu_mem_usage=True,
        ).to("cuda").eval()
        self._prefix_ids = self.tokenizer.encode(
            self._PREFIX, add_special_tokens=False
        )
        self._suffix_ids = self.tokenizer.encode(
            self._SUFFIX, add_special_tokens=False
        )
        no_ids = self.tokenizer.encode("no", add_special_tokens=False)
        yes_ids = self.tokenizer.encode("yes", add_special_tokens=False)
        if len(no_ids) != 1 or len(yes_ids) != 1:
            raise RuntimeError("Qwen reranker yes/no labels are not single tokens")
        self._label_ids = [no_ids[0], yes_ids[0]]

    def score(
        self, pairs: list[tuple[str, str]], *, instruction: str,
        max_length: int | None = None,
    ) -> list[float]:
        torch = self._torch
        output: list[float] = []
        active_max_length = int(max_length or self.max_length)
        body_budget = active_max_length - len(self._prefix_ids) - len(
            self._suffix_ids
        )
        if body_budget < 64:
            raise ValueError("reranker max_length leaves no body token budget")
        with torch.inference_mode():
            for start in _batch_progress(
                range(0, len(pairs), self.batch_size), "Qwen reranker pairs"
            ):
                batch_pairs = pairs[start:start + self.batch_size]
                records = []
                for query, document in batch_pairs:
                    body = (
                        f"<Instruct>: {instruction}\n"
                        f"<Query>: {query}\n"
                        f"<Document>: {document}"
                    )
                    body_ids = self.tokenizer.encode(
                        body,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=body_budget,
                    )
                    records.append({
                        "input_ids": [
                            *self._prefix_ids, *body_ids, *self._suffix_ids
                        ],
                    })
                encoded = self.tokenizer.pad(
                    records, padding=True, return_tensors="pt"
                ).to("cuda")
                logits = self.model(
                    **encoded, use_cache=False
                ).logits[:, -1, self._label_ids]
                probabilities = torch.softmax(logits.float(), dim=-1)[:, 1]
                output.extend(float(value) for value in probabilities.cpu())
        return output

    def close(self) -> None:
        model = getattr(self, "model", None)
        tokenizer = getattr(self, "tokenizer", None)
        if model is not None:
            del self.model
        if tokenizer is not None:
            del self.tokenizer
        gc.collect()
        self._torch.cuda.empty_cache()


def runtime_fingerprint(backend: str) -> dict:
    result = {
        "backend": backend,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {},
        "cuda_available": False,
        "gpus": [],
        "peak_gpu_memory_bytes": 0,
    }
    for name in ("numpy", "pandas", "pyarrow", "torch", "transformers"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    try:
        import torch
        result["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            result["gpus"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_bytes": int(
                        torch.cuda.get_device_properties(index).total_memory
                    ),
                }
                for index in range(torch.cuda.device_count())
            ]
            result["peak_gpu_memory_bytes"] = int(
                max(
                    torch.cuda.max_memory_allocated(index)
                    for index in range(torch.cuda.device_count())
                )
            )
    except ImportError:
        pass
    return result


def build_embedding_backend(
    backend: str, model_config: dict, runtime: dict
) -> EmbeddingBackend:
    if backend == "fake":
        return FakeEmbeddingBackend()
    if backend == "qwen":
        return QwenEmbeddingBackend(model_config, runtime)
    raise ValueError(f"unknown embedding backend: {backend}")


def build_reranker_backend(
    backend: str, model_config: dict, runtime: dict
) -> RerankerBackend:
    if backend == "fake":
        return FakeRerankerBackend()
    if backend == "qwen":
        return QwenRerankerBackend(model_config, runtime)
    raise ValueError(f"unknown reranker backend: {backend}")


def _require_cuda_runtime() -> None:
    try:
        import torch
        import transformers
    except ImportError as error:
        raise RuntimeError(
            "Qwen backend requires torch and transformers>=4.51.0"
        ) from error
    version = tuple(
        int(part) for part in transformers.__version__.split(".")[:2]
    )
    if version < (4, 51):
        raise RuntimeError(
            f"transformers {transformers.__version__} is too old; need >=4.51.0"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen backend requires a CUDA GPU")


def _last_token_pool(last_hidden_states, attention_mask):
    left_padding = (
        attention_mask[:, -1].sum() == attention_mask.shape[0]
    )
    if bool(left_padding):
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = attention_mask.new_tensor(
        range(last_hidden_states.shape[0])
    )
    return last_hidden_states[
        batch_indices, sequence_lengths,
    ]


def _batch_progress(iterable, description: str):
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=description, leave=False)
    except ImportError:
        return iterable
