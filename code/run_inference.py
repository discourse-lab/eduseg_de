#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inference script for EDU segmentation (German). Update the config in main() before use!

Supports loading the model from:
- Hugging Face Hub
- GitHub (raw files)

Model:
  discourse_lab / eduseg_de

"""

import os
from dataclasses import dataclass
from typing import List, Dict

import torch
from torch.utils.data import DataLoader
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
)

# =====================
# Label configuration
# =====================

label_list = ["I", "B"]
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

# =====================
# Configuration
# =====================

@dataclass
class InferenceConfig:
    model_source: str            # "hf" or "github"
    model_path: str
    input_dir: str
    output_dir: str
    max_length: int = 512
    batch_size: int = 4
    debug: bool = False


# =====================
# Data Collator
# =====================

class DataCollatorWithMeta:
    """
    Wraps HF's DataCollatorForTokenClassification
    while preserving filename, original_text, and offset_mapping.
    """

    def __init__(self, tokenizer):
        self.inner = DataCollatorForTokenClassification(tokenizer)

    def __call__(self, features: List[Dict]) -> Dict:
        filenames = [f["filename"] for f in features]
        texts = [f["original_text"] for f in features]
        offsets = [f["offset_mapping"] for f in features]

        tensor_features = [
            {k: v for k, v in f.items()
             if k not in {"filename", "original_text", "offset_mapping"}}
            for f in features
        ]

        batch = self.inner(tensor_features)
        batch["filename"] = filenames
        batch["original_text"] = texts
        batch["offset_mapping"] = offsets
        return batch


# =====================
# Dataset Preparation
# =====================

def prepare_inference_dataset(
    folder_path: str,
    tokenizer,
    max_length: int
) -> Dataset:
    """
    Tokenize all .txt files in a folder (no sliding window).
    Preserves offset mappings and metadata.
    """
    features = []

    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith(".txt"):
            continue

        with open(os.path.join(folder_path, fname), "r", encoding="utf-8") as f:
            text = f.read().strip()

        encodings = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )

        offset_map = [
            (int(s), int(e)) for s, e in encodings["offset_mapping"]
        ]

        features.append({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "offset_mapping": offset_map,
            "original_text": text,
            "filename": fname,
        })

    return Dataset.from_list(features)


# =====================
# EDU Reconstruction
# =====================

def reconstruct_edus_from_offsets(
    label_ids: List[int],
    offset_mapping: List,
    original_text: str,
) -> List[str]:
    """
    Reconstruct EDUs using BIO labels and offset mappings.
    """
    edus = []
    current_start, current_end = None, None
    text_len = len(original_text)

    for lab, (start, end) in zip(label_ids, offset_mapping):
        # skip padding & special tokens
        if lab == -100 or start == end:
            continue

        start = max(0, min(start, text_len))
        end = max(start, min(end, text_len))

        label = id2label[lab]

        if label.startswith("B") and current_start is not None:
            edus.append(original_text[current_start:current_end])
            current_start, current_end = start, end
        else:
            if current_start is None:
                current_start = start
            current_end = end

    if current_start is not None:
        edus.append(original_text[current_start:current_end])

    return edus


# =====================
# Inference
# =====================

def run_inference(
    model,
    tokenizer,
    config: InferenceConfig,
    device: torch.device,
):
    os.makedirs(config.output_dir, exist_ok=True)

    dataset = prepare_inference_dataset(
        config.input_dir,
        tokenizer,
        config.max_length,
    )

    collator = DataCollatorWithMeta(tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=collator,
    )

    model.eval()
    pred_by_file: Dict[str, List[str]] = {}

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            filenames = batch["filename"]
            offset_mappings = batch["offset_mapping"]
            original_texts = batch["original_text"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().tolist()

            if config.debug:
                from collections import Counter
                counter = Counter(preds[0])
                print("Label distribution:",
                      {id2label[k]: v for k, v in counter.items()})

            for fname, pred_seq, offset_map, orig_text in zip(
                filenames, preds, offset_mappings, original_texts
            ):
                edus = reconstruct_edus_from_offsets(
                    pred_seq,
                    offset_map,
                    orig_text,
                )
                pred_by_file.setdefault(fname, []).extend(edus)

    # Write output files
    for fname, edus in pred_by_file.items():
        out_path = os.path.join(config.output_dir, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            for edu in edus:
                edu = edu.strip()
                if edu:
                    f.write(edu + "\n")
        print(f"Saved predictions for {fname} ({len(edus)} EDUs)")


# =====================
# Main
# =====================

def main():

    # -------- Choose ONE model source --------

    # Hugging Face
    MODEL_PATH_HF = "sfrenzel/eduseg_de"

    # GitHub (raw files)
    MODEL_PATH_GH = (
        "https://github.com/discourse_lab/eduseg_de/raw/paper-v1.0/model/"
    )

    config = InferenceConfig(
        model_source="hf",          # "hf" or "github"
        model_path=MODEL_PATH_HF,   # switch to MODEL_PATH_GH if needed
        input_dir="inference_input",
        output_dir="inference_output",
        max_length=512,
        batch_size=4,
        debug=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    model = AutoModelForTokenClassification.from_pretrained(
        config.model_path
    ).to(device)

    # Safety check
    assert model.config.num_labels == len(label_list), (
        f"Model expects {model.config.num_labels} labels, "
        f"but label_list has {len(label_list)}"
    )

    run_inference(model, tokenizer, config, device)


if __name__ == "__main__":
    main()
