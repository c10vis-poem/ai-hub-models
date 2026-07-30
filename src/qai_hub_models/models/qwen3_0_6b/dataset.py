# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Self-generated calibration datasets for Qwen3-0.6B.

``GeneratedDataset`` calibrates on text the float model generated from seed
prompts ("output generated from the input"). The corpus is generated OFFLINE,
once, by ``generate_calibration_text`` (ported from Qualcomm GenAI Lab's
``GeneratedDataset._generate_encoded``) and stored as an asset; the class is
then a plain disk-backed text corpus, structurally identical to ``WikiText``.
Once the corpus exists no model is loaded and calibration stays deterministic
and cheap. If the corpus is missing from the asset store (e.g. a fresh branch),
``_download_data`` generates it on demand so quantize/eval are self-healing.

``InterleavedGeneratedWikitext`` round-robins ``GeneratedDataset`` with WikiText
for the final calibration pass. These live under the model dir (not the shared
``datasets/`` package) because qwen3_0_6b is the only model using this recipe.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedTokenizerBase,
    pipeline,
    set_seed,
)

from qai_hub_models.datasets.wikitext.wikitext import WikiText
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.base_dataset import (
    BaseDataset,
    DatasetMetadata,
    DatasetSplit,
    InterleavedDataset,
)

GENERATED_DATASET_ID = "generated_calibration_qwen3_0_6b"
GENERATED_DATASET_VERSION = 1
GENERATED_CORPUS_FILENAME = "generated_calibration.txt"

# Float model the corpus is generated from, and the default corpus size (in
# context-length blocks) to produce when auto-generating a missing corpus.
GENERATION_HF_REPO_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_GENERATION_NUM_BLOCKS = 12

# Seed prompts (vendored from GenAI Lab, disjoint from the eval prompt set).
SEED_PROMPTS_FILE = Path(__file__).parent / "calibration_prompts.yaml"
SYSTEM_PROMPT = "You are a helpful AI assistant."


def load_seed_prompts(prompts_file: Path = SEED_PROMPTS_FILE) -> list[str]:
    with open(prompts_file) as f:
        prompts = yaml.safe_load(f)
    assert isinstance(prompts, list) and len(prompts) > 0
    return prompts


def generate_calibration_text(
    hf_repo_name: str,
    num_blocks: int,
    context_length: int,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 40,
    seed: int = 42,
) -> str:
    """Generate calibration text from the float model and return it as a string.

    Cycles the seed prompts in order, generating prompt+completion text until the
    re-tokenized concatenation reaches ``num_blocks * context_length`` tokens,
    then truncates to exactly that many tokens so the corpus is a whole number of
    context-length chunks. "Output generated from the input": the seed prompts
    are the input, the model's completions are the calibration text.
    """
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(hf_repo_name)
    model = AutoModelForCausalLM.from_pretrained(
        hf_repo_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    prompts = load_seed_prompts()
    target_tokens = num_blocks * context_length

    # Merge EOS tokens from model config + tokenizer (mirrors GenAI Lab's
    # build_generation_config).
    eos_ids: set[int] = set()
    for src in (getattr(model.config, "eos_token_id", None), tokenizer.eos_token_id):
        if src is None:
            continue
        if isinstance(src, (list, tuple)):
            eos_ids.update(src)
        else:
            eos_ids.add(src)
    gen_config = GenerationConfig(
        eos_token_id=sorted(eos_ids) if eos_ids else tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    # Same join rule as WikiText: bos_token for TRAIN-style calibration data
    # reduces quantization accuracy drop vs. a plain "\n\n" separator.
    join_token = "\n\n" if tokenizer.bos_token is None else tokenizer.bos_token

    generated_texts: list[str] = []
    total_tokens = 0
    prompt_idx = 0
    # Hard cap guards against pathological early-EOS loops that never reach the
    # target (2x the prompts needed in the ideal case).
    max_generations = 2 * (target_tokens // max(max_new_tokens, 1) + len(prompts))
    progress = tqdm(
        total=target_tokens,
        unit="tok",
        desc=f"Generating calibration text ({hf_repo_name})",
    )
    try:
        while total_tokens < target_tokens and len(generated_texts) < max_generations:
            prompt = prompts[prompt_idx % len(prompts)]
            prompt_idx += 1
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            # return_full_text=True yields the full conversation as message
            # dicts; render it back to a single prompt+completion string.
            full_messages = generator(
                messages,
                generation_config=gen_config,
                return_full_text=True,
            )[0]["generated_text"]
            text = tokenizer.apply_chat_template(full_messages, tokenize=False)
            assert isinstance(text, str)
            generated_texts.append(text)
            total_tokens = len(
                tokenizer(
                    join_token.join(generated_texts),
                    add_special_tokens=True,
                )["input_ids"]
            )
            progress.n = min(total_tokens, target_tokens)
            progress.refresh()
    finally:
        progress.close()
        del generator
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    corpus = join_token.join(generated_texts)
    # Truncate to exactly target_tokens (a whole number of context-length chunks)
    # by tokenizing, slicing, and decoding back.
    encoded = tokenizer(corpus, add_special_tokens=True)["input_ids"]
    if len(encoded) > target_tokens:
        encoded = encoded[:target_tokens]
        corpus = tokenizer.decode(encoded, skip_special_tokens=False)
    return corpus


class GeneratedDataset(BaseDataset):
    """Calibrate on text the float model generated offline from seed prompts."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        block_size: int = 128,
        context_length: int = 4096,
        split: DatasetSplit = DatasetSplit.TRAIN,
        num_samples: int = 0,
        dataset_id: str = GENERATED_DATASET_ID,
        dataset_version: int | str = GENERATED_DATASET_VERSION,
        corpus_filename: str = GENERATED_CORPUS_FILENAME,
    ) -> None:
        self.block_size = block_size
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.num_samples = num_samples

        self._corpus_asset = CachedWebDatasetAsset.from_asset_store(
            dataset_id, dataset_version, corpus_filename
        )
        super().__init__(self._corpus_asset.path, split)
        with open(self._corpus_asset.path) as f:
            corpus = f.read()

        # Same join/tokenize convention as WikiText's TRAIN split (the corpus is
        # already a single joined string, so we just tokenize it directly).
        self.tokens = self.tokenizer(
            corpus,
            return_tensors="pt",
            add_special_tokens=True,
        )

    @staticmethod
    def collate_fn(
        batch: list[dict[str, torch.Tensor]],
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ]:
        return (
            batch[0]["input_ids"],
            batch[0]["attention_mask"],
            batch[0].get("label", batch[0]["input_ids"]),
        )

    def __len__(self) -> int:
        max_num = len(self.tokens["input_ids"][0]) // self.context_length
        num = self.num_samples if self.num_samples != 0 else max_num
        return min(num, max_num)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        num_tokens = self.tokens["input_ids"].shape[-1]
        start_index = index * self.context_length
        end_index = min((index + 1) * self.context_length, num_tokens)
        return {
            "input_ids": self.tokens["input_ids"][:, start_index:end_index],
            "attention_mask": self.tokens["attention_mask"][:, start_index:end_index],
        }

    def _download_data(self) -> None:
        # Prefer the published asset; if it isn't in the store yet (e.g. a fresh
        # branch that hasn't uploaded the corpus), generate it offline from the
        # float model so quantize/eval are self-healing. Generated once, then
        # cached at the asset path for subsequent runs.
        try:
            self._corpus_asset.fetch()
            return
        except Exception:
            pass

        print(
            "Generated calibration corpus not found in the asset store; "
            "generating it offline from the float model (one-time)."
        )
        # Produce enough blocks to cover this dataset's need with margin.
        num_blocks = max(DEFAULT_GENERATION_NUM_BLOCKS, self.num_samples)
        corpus = generate_calibration_text(
            hf_repo_name=GENERATION_HF_REPO_NAME,
            num_blocks=num_blocks,
            context_length=self.context_length,
        )
        corpus_path = self._corpus_asset.path
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        with open(corpus_path, "w") as f:
            f.write(corpus)

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 1

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="",
            split_description="Self-generated calibration corpus (offline generated from seed prompts)",
        )

    @classmethod
    def dataset_name(cls) -> str:
        return "generated"


class InterleavedGeneratedWikitext(InterleavedDataset):
    """Interleaves self-generated calibration text and Wikitext for LLM calibration."""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        num_samples: int = 0,
        **kwargs: Any,
    ) -> None:
        # InterleavedDataset.__init__ binds num_samples as a named param, so it
        # is stripped from **kwargs before load_datasets runs. Capture it here
        # so load_datasets can split it across the two sources.
        self._requested_num_samples = num_samples
        super().__init__(split=split, num_samples=num_samples, **kwargs)

    @classmethod
    def dataset_name(cls) -> str:
        return "interleaved_generated_wikitext"

    def load_datasets(self, split: DatasetSplit, **kwargs: Any) -> list[BaseDataset]:
        per_source = self._requested_num_samples // 2
        return [
            GeneratedDataset(
                split=split,
                tokenizer=kwargs["tokenizer"],
                block_size=kwargs.get("block_size", 128),
                context_length=kwargs.get("context_length", 4096),
                num_samples=per_source,
            ),
            WikiText(
                split=split,
                tokenizer=kwargs["tokenizer"],
                block_size=kwargs.get("block_size", 128),
                context_length=kwargs.get("context_length", 4096),
                num_samples=per_source,
            ),
        ]
