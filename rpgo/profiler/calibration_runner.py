"""CalibrationRunner: drives the routing profiler over a dataset.

Handles model loading, dataset preparation, and batched forward passes
for calibration. This is the main entry point for Phase 1 of R-PGO.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from rpgo._core import RoutingProfile
from rpgo.hf_compat import patch_transformers_remote_code_compat
from rpgo.profiler.routing_profiler import RoutingProfiler

logger = logging.getLogger(__name__)


class CalibrationRunner:
    """Runs calibration forward passes to collect routing profiles.

    Usage:
        runner = CalibrationRunner(
            model_id="Qwen/Qwen3-30B-A3B",
            n_samples=2048,
            max_length=512,
        )
        profile = runner.run()
        profile.save("routing_profile.json")
    """

    def __init__(
        self,
        model_id: str,
        n_samples: int = 2048,
        max_length: int = 512,
        batch_size: int = 1,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-103-raw-v1",
        dataset_split: str = "train",
        device: str | None = None,
        torch_dtype: Any = None,
        load_in_4bit: bool = False,
    ):
        self.model_id = model_id
        self.n_samples = n_samples
        self.max_length = max_length
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.dataset_split = dataset_split
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or torch.float16
        self.load_in_4bit = load_in_4bit

    def run(self, output_path: str | Path | None = None) -> RoutingProfile:
        """Run the full calibration pipeline.

        1. Load model + tokenizer
        2. Load dataset
        3. Run profiling forward passes
        4. Build and optionally save the routing profile

        Requires: transformers, datasets (install with `pip install rpgo[profile]`)
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Calibration requires 'transformers' and 'datasets'. "
                "Install with: pip install rpgo[profile]"
            )

        patch_transformers_remote_code_compat()
        logger.info(f"Loading model: {self.model_id}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        offload_folder = os.environ.get("RPGO_OFFLOAD_DIR") or str(
            Path(tempfile.gettempdir()) / "rpgo_offload"
        )
        Path(offload_folder).mkdir(parents=True, exist_ok=True)

        load_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "trust_remote_code": True,
            "offload_folder": offload_folder,
        }
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            # Use device_map="auto" to spill large models to CPU when needed
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
            )
        else:
            load_kwargs["dtype"] = self.torch_dtype

        model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
        model.eval()

        logger.info(f"Loading dataset: {self.dataset_name}/{self.dataset_config}")
        dataset = load_dataset(
            self.dataset_name,
            self.dataset_config,
            split=self.dataset_split,
        )

        # Tokenize
        def tokenize(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )

        logger.info(f"Running calibration: {self.n_samples} samples")
        profiler = RoutingProfiler(model, model_id=self.model_id)
        profiler.start()

        samples_processed = 0
        with torch.no_grad():
            for i in range(0, min(len(dataset), self.n_samples), self.batch_size):
                batch_texts = []
                for j in range(self.batch_size):
                    if i + j < len(dataset) and samples_processed < self.n_samples:
                        text = dataset[i + j].get("text", "")
                        if text.strip():
                            batch_texts.append(text)
                            samples_processed += 1

                if not batch_texts:
                    continue

                inputs = tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)

                model(**inputs, use_cache=False)

                if samples_processed % 100 == 0:
                    logger.info(f"  Processed {samples_processed}/{self.n_samples} samples")

                if samples_processed >= self.n_samples:
                    break

        profiler.stop()

        logger.info(f"Calibration complete: {samples_processed} samples, "
                     f"{profiler._total_tokens} total token activations")

        profile = profiler.build_profile(calibration_samples=samples_processed)

        if output_path is not None:
            profile.save(str(output_path))
            logger.info(f"Profile saved to {output_path}")

        return profile


def run_calibration_from_config(config: dict[str, Any]) -> RoutingProfile:
    """Run calibration from a YAML config dict."""
    runner = CalibrationRunner(
        model_id=config["model_id"],
        n_samples=config.get("calibration_samples", 2048),
        max_length=config.get("max_length", 512),
        batch_size=config.get("batch_size", 1),
        dataset_name=config.get("dataset_name", "wikitext"),
        dataset_config=config.get("dataset_config", "wikitext-103-raw-v1"),
        dataset_split=config.get("dataset_split", "train"),
        load_in_4bit=config.get("load_in_4bit", False),
    )
    output_path = config.get("output_path", "routing_profile.json")
    return runner.run(output_path=output_path)
