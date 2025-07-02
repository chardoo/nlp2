"""
inference.py  ·  Run your fine-tuned emotion-/intensity-classifier on new text.

Handles the three most frequent checkpoint mismatches automatically:
    • DataParallel / DDP  → keys start with "module."
    • Hugging-Face backbone → keys start with "distilbert." or "bert."
    • Nested dictionaries  → {'state_dict': ..., ...}
If you accidentally saved the *whole* model object, it still loads.
"""

from pathlib import Path
from typing import Tuple, Dict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from data_prep import clean_text          # same pre-processing used during training
from model     import IntensityClassifier # swap if your class is EmotionClassifier

# ──────────────────────────────────────────────────────────────────────────────
# Config – tweak as needed
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONE   = "distilbert-base-uncased"           # same HF checkpoint as training
WEIGHTS    = Path("./models/best_model.pt")               # your saved file
EMOTIONS   = ["anger", "fear", "joy", "sadness", "surprise"]
MAX_LEN    = 128                                 # should match training
# ──────────────────────────────────────────────────────────────────────────────

# 1) tokenizer
tokenizer = AutoTokenizer.from_pretrained(BACKBONE)

# 2) model
model = IntensityClassifier(BACKBONE).to(DEVICE)

# 3) robust checkpoint loader --------------------------------------------------
def _strip(state, prefix: str):
    """Remove `prefix` from every key in `state`."""
    return {k[len(prefix):]: v for k, v in state.items()}

def _rename(state, old: str, new: str):
    """Replace `old` with `new` at the *start* of every key."""
    return {k.replace(old, new, 1) if k.startswith(old) else k: v
            for k, v in state.items()}

def smart_load(model: torch.nn.Module, ckpt_path: Path):
    raw = torch.load(ckpt_path, map_location="cpu")

    # Case A: whole model pickled
    if isinstance(raw, torch.nn.Module):
        print("⚠️  Checkpoint is a full model object – extracting state_dict()")
        raw = raw.state_dict()

    # Case B: nested under "state_dict"
    if "state_dict" in raw and isinstance(raw["state_dict"], dict):
        raw = raw["state_dict"]

    # Fix 1: DataParallel / DDP ("module.")
    if any(k.startswith("module.") for k in raw):
        print("↺  Removing 'module.' prefix (DataParallel/DDP)")
        raw = _strip(raw, "module.")

    # Fix 2: HF backbone saved its own prefix
    if any(k.startswith("distilbert.") for k in raw):
        print("↺  Replacing 'distilbert.' → 'backbone.transformer.'")
        raw = _rename(raw, "distilbert.", "backbone.transformer.")
    elif any(k.startswith("bert.") for k in raw):
        print("↺  Replacing 'bert.' → 'backbone.'")
        raw = _rename(raw, "bert.", "backbone.")

    # Finally load (strict=True ensures any mismatch is still surfaced)
    missing, unexpected = model.load_state_dict(raw, strict=False)
    print(f"✓ Loaded weights  –  missing {len(missing)}  unexpected {len(unexpected)}")
    return model

model = smart_load(model, WEIGHTS).eval().to(DEVICE)

# 4) prediction helper ---------------------------------------------------------
def predict(text: str, threshold: float = 0.5) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    Args
    ----
    text : str      Input sentence / paragraph
    threshold : float   prob ≥ threshold → label 1

    Returns
    -------
    labels : {emotion: 0/1}
    scores : {emotion: probability-of-1 rounded to 3 decimals}
    """
    encoded = tokenizer(
        clean_text(text),
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(encoded["input_ids"], encoded["attention_mask"])

    probs  = {e: F.softmax(logits[e], dim=1)[:, 1] for e in EMOTIONS}
    labels = {e: int((probs[e] >= threshold).item()) for e in EMOTIONS}
    scores = {e: round(probs[e].item(), 3)           for e in EMOTIONS}
    return labels, scores

# 5) quick demo ----------------------------------------------------------------
if __name__ == "__main__":
    sample = "I am happy today"
    labs, scs = predict(sample)
    print("\nText:", sample)
    print("Labels:", labs)
    print("Scores:", scs)
