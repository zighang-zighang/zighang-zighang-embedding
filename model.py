import logging
import numpy as np
import torch
from dataclasses import dataclass
from enum import Enum
from transformers import AutoModel, AutoTokenizer
from typing import Optional

logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    MEAN = "mean"
    MAX = "max"
    WEIGHTED = "weighted"


@dataclass
class EmbeddingConfig:
    model_name: str = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
    max_length: int = 1024
    normalize: bool = True
    chunk_overlap: int = 100
    aggregation_method: AggregationMethod = AggregationMethod.MEAN


class KoreanEmbedding:
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        logger.info(f"Initializing KoreanEmbedding with model: {self.config.model_name}")

        self.device = self._get_optimal_device()
        logger.info(f"Using device: {self.device}")

        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        logger.info("Loading model...")
        self.model = AutoModel.from_pretrained(
            self.config.model_name,
            add_pooling_layer=False
        ).to(self.device)
        self.model.eval()
        logger.info("Model initialization completed")

    @staticmethod
    def _get_optimal_device() -> str:
        if torch.cuda.is_available():
            device = "cuda"
            logger.debug(f"CUDA available with {torch.cuda.device_count()} device(s)")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            logger.debug("MPS (Apple Silicon) available")
        else:
            device = "cpu"
            logger.debug("Using CPU device")

        return device

    def _chunk_text(self, text: str) -> list[str]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= self.config.max_length - 2:
            return [text]

        chunks = []
        chunk_size = self.config.max_length - 2 - self.config.chunk_overlap

        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + self.config.max_length - 2]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)

        return chunks

    def _aggregate_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        method = self.config.aggregation_method

        if method == AggregationMethod.MEAN:
            return np.mean(embeddings, axis=0)
        elif method == AggregationMethod.MAX:
            return np.max(embeddings, axis=0)
        elif method == AggregationMethod.WEIGHTED:
            weights = np.exp(-np.arange(len(embeddings)) * 0.1)
            weights = weights / np.sum(weights)

            return np.average(embeddings, axis=0, weights=weights)

    def encode(self, text: str) -> np.ndarray:
        token_count = len(self.tokenizer.encode(text, add_special_tokens=True))

        if token_count > self.config.max_length:
            chunks = self._chunk_text(text)
            chunk_embeddings = self._encode_chunks(chunks)
            result = self._aggregate_embeddings(chunk_embeddings)
            return result
        else:
            return self._encode_single_text(text)

    def _encode_chunks(self, chunks: list[str]) -> np.ndarray:
        encoded_input = self.tokenizer(
            chunks,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )

        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embeddings = model_output[0][:, 0]

            if self.config.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    def _encode_single_text(self, text: str) -> np.ndarray:
        encoded_input = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )

        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embedding = model_output[0][0, 0]

            if self.config.normalize:
                embedding = torch.nn.functional.normalize(embedding.unsqueeze(0), p=2, dim=1)[0]

        return embedding.cpu().numpy()


if __name__ == "__main__":
    import os
    import glob
    from pathlib import Path
    from tqdm import tqdm

    logging.basicConfig(level=logging.INFO)

    input_dir = "input"
    output_dir = "output"

    os.makedirs(output_dir, exist_ok=True)

    embedding_model = KoreanEmbedding()

    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))

    for txt_file in tqdm(txt_files, desc="Processing files", unit="file"):
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        embedding = embedding_model.encode(text)

        filename = Path(txt_file).stem
        output_file = os.path.join(output_dir, f"{filename}.npy")

        np.save(output_file, embedding)
