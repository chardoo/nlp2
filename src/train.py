
# ---------- Training & Evaluation ----------



from sklearn.metrics import f1_score
from transformers import AutoTokenizer
import torch
from torch.utils.data import TensorDataset

from data_prep import clean_text
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
from torch.optim import AdamW


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for input_ids, attn_mask, labels in loader:
        input_ids, attn_mask, labels = input_ids.to(device), attn_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attn_mask)

        # Sum cross-entropy over heads
        loss = 0
        for i, emo in enumerate(logits):
            loss += criterion(logits[emo], labels[:, i])
        loss = loss / len(logits)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, attn_mask, labels in loader:
            input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
            logits = model(input_ids, attn_mask)
            # Argmax per head
            preds = torch.stack([logits[emo].argmax(dim=1) for emo in logits], dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    return preds, labels


def calculate_multilabel_f1(labels, preds):
    """
    Calculate F1 score for multi-label classification.
    """
    emotion_names = ["anger", "fear", "joy", "sadness", "surprise"]
    f1_scores = []
    
    # Calculate F1 for each emotion separately
    for i in range(len(emotion_names)):
        emotion_labels = labels[:, i]
        emotion_preds = preds[:, i]
        
        # Calculate F1 for this emotion
        emotion_f1 = f1_score(emotion_labels, emotion_preds, average='macro')
        f1_scores.append(emotion_f1)
        print(f"  {emotion_names[i]}: F1={emotion_f1:.4f}")
    
    # Return macro average across all emotions
    macro_f1 = sum(f1_scores) / len(f1_scores)
    return macro_f1