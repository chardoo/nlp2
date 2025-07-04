"""
Multi-label Emotion Intensity Classification using RoBERTa

This module implements a deep learning model for classifying emotion intensity
across five emotions (anger, fear, joy, sadness, surprise) with intensity levels 0-3.
"""

import os
import re
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score

from data_prep import clean_text
from evaluate import calculate_multilabel_f1_score, evaluate_epoch, train_epoch


# Constants
# PRETRAINED_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
PRETRAINED_MODEL =   "distilbert-base-uncased"

EMOTION_LABELS = ["anger", "fear", "joy", "sadness", "surprise"]
MAX_SEQUENCE_LENGTH = 128
NUM_INTENSITY_CLASSES = 4


def tokenize_texts(texts):
    """
    Tokenize input texts using sentiment-analysis BERT tokenizer.
    
    Args:
        texts: List of text strings to tokenize
        
    Returns:
        Dictionary containing tokenized encodings with input_ids and attention_mask
    """
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        return_tensors="pt"
    )
    return encodings


def create_tensor_dataset(encodings, labels):
    """
    Create a TensorDataset from tokenized encodings and labels.
    
    Args:
        encodings: Dictionary containing input_ids and attention_mask tensors
        labels: Tensor of emotion intensity labels
        
    Returns:
        TensorDataset object ready for DataLoader
    """
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    return TensorDataset(input_ids, attention_mask, labels)


def prepare_dataset(data_path):
    """
    Prepare dataset for multi-label emotion intensity classification.
    Each emotion has intensity values from 0-3.
    
    Args:
        data_path: Path to the CSV file containing the dataset
        
    Returns:
        TensorDataset ready for training
    """
    clean_data = pd.read_csv(data_path, encoding='unicode_escape')
    texts = clean_data["text"].tolist()
    encodings = tokenize_texts(texts)
    
    # Convert emotion intensity labels (0-3) to long tensor for CrossEntropyLoss
    labels = torch.tensor(clean_data[EMOTION_LABELS].values, dtype=torch.long)
    
    # Verify label ranges are correct (0-3)
    print(f"Label ranges: min={labels.min().item()}, max={labels.max().item()}")
    assert labels.min() >= 0 and labels.max() <= 3, "Labels must be in range 0-3"
    
    tensor_dataset = create_tensor_dataset(encodings, labels)
    print(f"TensorDataset created with {len(tensor_dataset)} samples.")
    return tensor_dataset


def load_data(data_path, batch_size=16):
    """
    Load and split data into training and validation sets.
    
    Args:
        data_path: Path to the dataset CSV file
        batch_size: Batch size for DataLoaders
        
    Returns:
        Tuple of (train_loader, validation_loader)
    """
    tensor_dataset = prepare_dataset(data_path)
    total_size = len(tensor_dataset)
    validation_size = int(0.1 * total_size)
    train_size = total_size - validation_size

    train_dataset, validation_dataset = random_split(tensor_dataset, [train_size, validation_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
    return train_loader, validation_loader


class EmotionIntensityClassifier(nn.Module):
    """
    Multi-label emotion intensity classifier using RoBERTa backbone.
    
    This model classifies text into 5 emotions with intensity levels 0-3 for each emotion.
    """
    
    def __init__(self, pretrained_model=PRETRAINED_MODEL, num_classes=NUM_INTENSITY_CLASSES):
        """
        Initialize the emotion intensity classifier.
        
        Args:
            pretrained_model: Name of the pretrained model to use as backbone
            num_classes: Number of intensity classes (default: 4 for levels 0-3)
        """
        super().__init__()
        # Using sentiment-analysis specific model for better emotion understanding
        self.backbone = AutoModel.from_pretrained(pretrained_model)
        hidden_size = self.backbone.config.hidden_size
        self.emotion_names = EMOTION_LABELS
        
        # Create separate classification heads for each emotion with dropout for regularization
        self.dropout = nn.Dropout(0.3)
        self.classification_heads = nn.ModuleDict({
            emotion: nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size//2, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size//4, hidden_size // 6),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size // 6, num_classes)
            )
            for emotion in self.emotion_names
        })

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs tensor of shape [batch_size, sequence_length]
            attention_mask: Attention mask tensor of shape [batch_size, sequence_length]
            
        Returns:
            Dictionary mapping emotion names to logits tensors
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_representation = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        cls_representation = self.dropout(cls_representation)
        
        # Get logits for each emotion
        logits = {emotion: head(cls_representation) for emotion, head in self.classification_heads.items()}
        return logits


def train_epoch_example(model, train_loader, optimizer, criterion, device):
    """
    Training epoch for multi-label emotion intensity classification.
    Each emotion is classified independently into 4 intensity levels (0-3).
    
    Args:
        model: The emotion intensity classifier model
        train_loader: DataLoader for training data
        optimizer: Optimizer for updating model parameters
        criterion: Loss function (CrossEntropyLoss)
        device: Device to run computation on
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        input_ids, attention_mask, labels = [tensor.to(device) for tensor in batch]
        
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        
        # Calculate loss for each emotion separately
        # labels shape: [batch_size, 5] where each column is emotion intensity (0-3)
        batch_loss = 0
        for i, emotion in enumerate(model.emotion_names):
            emotion_labels = labels[:, i]  # Shape: [batch_size] with values 0-3
            emotion_logits = logits[emotion]  # Shape: [batch_size, 4] (4 intensity classes)
            batch_loss += criterion(emotion_logits, emotion_labels)
        
        # Average the loss across emotions
        batch_loss = batch_loss / len(model.emotion_names)
        
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()
    
    return total_loss / len(train_loader)


def evaluate_epoch_example(model, validation_loader, device):
    """
    Evaluation epoch for multi-label emotion intensity classification.
    Returns predictions and true labels for each emotion.
    
    Args:
        model: The emotion intensity classifier model
        validation_loader: DataLoader for validation data
        device: Device to run computation on
        
    Returns:
        Tuple of (all_predictions, all_labels) as dictionaries
    """
    model.eval()
    all_predictions = {emotion: [] for emotion in model.emotion_names}
    all_labels = {emotion: [] for emotion in model.emotion_names}
    
    with torch.no_grad():
        for batch in validation_loader:
            input_ids, attention_mask, labels = [tensor.to(device) for tensor in batch]
            logits = model(input_ids, attention_mask)
            
            for i, emotion in enumerate(model.emotion_names):
                # Get predicted intensity class (0-3)
                predictions = torch.argmax(logits[emotion], dim=1)
                all_predictions[emotion].extend(predictions.cpu().numpy())
                all_labels[emotion].extend(labels[:, i].cpu().numpy())
    
    return all_predictions, all_labels


def calculate_multilabel_f1_example(all_labels, all_predictions):
    """
    Calculate macro F1 score across all emotions.
    For intensity classification, we treat it as multi-class classification for each emotion.
    
    Args:
        all_labels: Dictionary mapping emotion names to lists of true labels
        all_predictions: Dictionary mapping emotion names to lists of predictions
        
    Returns:
        Overall macro F1 score
    """
    f1_scores = []
    
    for emotion in EMOTION_LABELS:
        f1 = f1_score(all_labels[emotion], all_predictions[emotion], average='macro')
        f1_scores.append(f1)
        print(f"{emotion.capitalize()} F1: {f1:.4f}")
    
    macro_f1 = sum(f1_scores) / len(f1_scores)
    print(f"Overall Macro F1: {macro_f1:.4f}")
    return macro_f1


def main():
    """Main training function."""
    # Paths & hyperparameters
    data_path = os.path.join('..', 'data', 'processed', 'track-b-clean.csv')
    num_epochs = 14
    learning_rate = 1e-5  # Slightly lower learning rate for sentiment-analysis models
    batch_size = 16
    models_directory = 'models'
    os.makedirs(models_directory, exist_ok=True)
    
    # Load data
    train_loader, validation_loader = load_data(data_path, batch_size=batch_size)

    # Device & model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmotionIntensityClassifier().to(device)

    # Optimizer & loss - Updated with weight decay for better regularization
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    best_f1_score = 0
    best_model_path = None

    print("Starting training...")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print("-" * 50)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        # Use your existing functions or the example ones above
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        predictions, labels = evaluate_epoch(model, validation_loader, device)
        
        f1 = calculate_multilabel_f1_score(labels, predictions)
        print(f"Epoch {epoch}: TrainLoss={train_loss:.4f}, ValMacroF1={f1:.4f}")

        # Save best model
        if f1 > best_f1_score:
            best_f1_score = f1
            best_model_path = os.path.join(models_directory, 'best_emotion_intensity_model.pt')
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ Best model saved to: {best_model_path}")
            
    print("Training completed!")
    print(f"Best F1 score: {best_f1_score:.4f}")
    print(f"Model saved as: {best_model_path}")


if __name__ == "__main__":
    main()