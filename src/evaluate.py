"""
Training and Evaluation Functions for Multi-label Emotion Intensity Classification

This module provides the core training and evaluation functions for emotion
intensity classification models.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

from data_prep import clean_text


# Constants
EMOTION_LABELS = ["anger", "fear", "joy", "sadness", "surprise"]


def train_epoch(model: nn.Module,
                data_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                device: torch.device) -> float:
    """
    Perform one training epoch for multi-label emotion intensity classification.
    
    Args:
        model: The emotion intensity classifier model
        data_loader: DataLoader containing training data
        optimizer: Optimizer for updating model parameters
        criterion: Loss function (typically CrossEntropyLoss)
        device: Device to run computation on (CPU or CUDA)
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    for input_ids, attention_mask, labels in data_loader:
        # Move tensors to device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        emotion_logits = model(input_ids, attention_mask)

        # Calculate loss for each emotion head
        batch_loss = 0.0
        for i, emotion in enumerate(emotion_logits):
            emotion_loss = criterion(emotion_logits[emotion], labels[:, i])
            batch_loss += emotion_loss
        
        # Average loss across all emotion heads
        batch_loss = batch_loss / len(emotion_logits)

        # Backward pass and optimization
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()
    
    return total_loss / len(data_loader)


def evaluate_epoch(model: nn.Module,
                   data_loader: DataLoader,
                   device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform one evaluation epoch for multi-label emotion intensity classification.
    
    Args:
        model: The emotion intensity classifier model
        data_loader: DataLoader containing evaluation data
        device: Device to run computation on (CPU or CUDA)
        
    Returns:
        Tuple of (predictions, true_labels) where both are tensors of shape
        [num_samples, num_emotions]
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for input_ids, attention_mask, labels in data_loader:
            # Move tensors to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Forward pass
            emotion_logits = model(input_ids, attention_mask)
            
            # Get predictions by taking argmax for each emotion head
            emotion_predictions = torch.stack([
                emotion_logits[emotion].argmax(dim=1) 
                for emotion in emotion_logits
            ], dim=1)
            
            # Store predictions and labels
            all_predictions.append(emotion_predictions.cpu())
            all_labels.append(labels)
    
    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0)
    true_labels = torch.cat(all_labels, dim=0)
    
    return predictions, true_labels


def calculate_multilabel_f1_score(true_labels: torch.Tensor,
                                  predictions: torch.Tensor) -> float:
    """
    Calculate macro F1 score for multi-label emotion classification.
    
    Args:
        true_labels: True labels tensor of shape [num_samples, num_emotions]
        predictions: Predicted labels tensor of shape [num_samples, num_emotions]
        
    Returns:
        Macro F1 score averaged across all emotions
    """
    f1_scores = []
    
    print("Individual emotion F1 scores:")
    # Calculate F1 for each emotion separately
    for i, emotion_name in enumerate(EMOTION_LABELS):
        emotion_true_labels = true_labels[:, i].numpy()
        emotion_predictions = predictions[:, i].numpy()
        
        # Calculate macro F1 for this emotion
        emotion_f1 = f1_score(
            emotion_true_labels, 
            emotion_predictions, 
            average='macro',
            zero_division=0
        )
        f1_scores.append(emotion_f1)
        print(f"  {emotion_name.capitalize()}: F1={emotion_f1:.4f}")
    
    # Calculate macro average across all emotions
    macro_f1_score = sum(f1_scores) / len(f1_scores)
    return macro_f1_score