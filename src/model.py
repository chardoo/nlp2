from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import torch
from torch.utils.data import TensorDataset
import re
import pandas as pd
import os
from data_prep import clean_text
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import f1_score
from train import calculate_multilabel_f1, eval_epoch, train_epoch
import numpy as np

# Provided configuration options
lr_options = 5e-5
batch_size_options = 32  # Start conservative, increase if memory allows
max_length_options = 256

def analyze_text_lengths(texts):
    """
    Analyze text lengths to determine optimal max_length.
    """
    lengths = [len(text.split()) for text in texts]
    print(f"Text length stats:")
    print(f"  Mean: {np.mean(lengths):.1f}")
    print(f"  Median: {np.median(lengths):.1f}")
    print(f"  95th percentile: {np.percentile(lengths, 95):.1f}")
    print(f"  Max: {max(lengths)}")
    
    # Recommend max_length based on 95th percentile
    recommended = min(512, max(128, int(np.percentile(lengths, 95) * 1.2)))
    print(f"  Recommended max_length: {recommended}")
    return recommended

def word_tokenizer(texts, max_length=256):
    """
    Tokenises the input text into a list of words with optimized max_length.
    """
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    encodings = tokenizer(
        texts,
        padding="max_length",    
        truncation=True,         
        max_length=max_length,   # Increased from 128
        return_tensors="pt"      
    )
    return encodings

def create_tensor_dataset(encodings, labels):
    """
    Converts encodings into a TensorDataset.
    """
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    return TensorDataset(input_ids, attention_mask, labels)

def prepare_dataset(data_path, max_length=256):
    """
    Prepare dataset for multi-label emotion intensity classification.
    Each emotion has intensity values from 0-3.
    """
    clean_data = pd.read_csv(data_path, encoding='unicode_escape')
    texts = clean_data["text"].tolist()
    
    # Analyze text lengths and get recommendation
    recommended_length = analyze_text_lengths(texts)
    actual_max_length = max_length if max_length else recommended_length
    print(f"Using max_length: {actual_max_length}")
    
    encoding = word_tokenizer(texts, max_length=actual_max_length)
    
    # Convert emotion intensity labels (0-3) to long tensor for CrossEntropyLoss
    emotion_cols = ["anger", "fear", "joy", "sadness", "surprise"]
    labels = torch.tensor(clean_data[emotion_cols].values, dtype=torch.long)
    
    # Verify label ranges are correct (0-3)
    print(f"Label ranges: min={labels.min().item()}, max={labels.max().item()}")
    assert labels.min() >= 0 and labels.max() <= 3, "Labels must be in range 0-3"
    
    tensor_dataset_obj = create_tensor_dataset(encoding, labels)
    print("TensorDataset created with {} samples.".format(len(tensor_dataset_obj)))
    return tensor_dataset_obj
  
def load_data(data_path, batch_size=32, max_length=256):
    tensorDataset = prepare_dataset(data_path, max_length)
    total_size = len(tensorDataset)
    val_size = int(0.15 * total_size)  # Increased validation set to 15%     
    train_size = total_size - val_size

    train_ds, val_ds = random_split(tensorDataset, [train_size, val_size])   

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size) 
    return train_loader, val_loader

class IntensityClassifier(nn.Module):
    def __init__(self, pretrained_model="google-bert/bert-base-uncased", num_classes=4, dropout_rate=0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(pretrained_model)
        hidden = self.backbone.config.hidden_size
        self.emotion_names = ["anger", "fear", "joy", "sadness", "surprise"]
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Create separate classification heads for each emotion with dropout
        self.heads = nn.ModuleDict({
            emo: nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden // 2, num_classes)
            )
            for emo in self.emotion_names
        })

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # [CLS] token
        cls = self.dropout(cls)
        
        # Get logits for each emotion
        logits = {emo: head(cls) for emo, head in self.heads.items()}
        return logits

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

def train_epoch_optimized(model, train_loader, optimizer, scheduler, criterion, device):
    """
    Optimized training epoch with gradient clipping and better loss handling.
    """
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        
        # Calculate loss for each emotion separately
        loss = 0
        for i, emo in enumerate(model.emotion_names):
            emo_labels = labels[:, i]  # Shape: [batch_size] with values 0-3
            emo_logits = logits[emo]   # Shape: [batch_size, 4] (4 intensity classes)
            loss += criterion(emo_logits, emo_labels)
        
        # Average the loss across emotions
        loss = loss / len(model.emotion_names)
        
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()  # Update learning rate
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def hyperparameter_search(data_path, models_dir):
    """
    Perform basic hyperparameter search for learning rate and batch size.
    """
    # Use provided configuration
    lr = lr_options
    batch_size = batch_size_options
    max_length = max_length_options
    
    best_config = None
    print("Using configuration: lr={}, batch_size={}, max_length={}".format(lr, batch_size, max_length))
    try:
        # Quick training for 3 epochs to evaluate
        config = {
            'lr': lr,
            'batch_size': batch_size,
            'max_length': max_length,
            'epochs': 3  # Quick evaluation
        }
        
        best_config = config
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"OOM error - skipping this configuration")
    
    
    return best_config

def train_with_config(data_path, config, models_dir, save_model=True):
    """
    Train model with specific configuration.
    """
    # Load data
    train_loader, val_loader = load_data(
        data_path, 
        batch_size=config['batch_size'], 
        max_length=config['max_length']
    )

    # Device & model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = IntensityClassifier(dropout_rate=0.3).to(device)

    # Optimizer with weight decay
    optimizer = AdamW(
        model.parameters(), 
        lr=config['lr'],
        weight_decay=0.01  # L2 regularization
    )
    
    # Learning rate scheduler with warmup
    num_training_steps = len(train_loader) * config['epochs']
    num_warmup_steps = num_training_steps // 10  # 10% warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=7, min_delta=0.001)

    best_f1 = 0
    
    # Training loop
    for epoch in range(1, config['epochs'] + 1):
        train_loss = train_epoch_optimized(model, train_loader, optimizer, scheduler, criterion, device)
        preds, labels = eval_epoch(model, val_loader, device)
        
        f1 = calculate_multilabel_f1(labels, preds)
        print(f"Epoch {epoch}: TrainLoss={train_loss:.4f}, ValMacroF1={f1:.4f}, LR={scheduler.get_last_lr()[0]:.2e}")

        # Early stopping check
        early_stopping(f1)
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            if save_model:
                model_path = os.path.join(models_dir, 'best_model.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'f1_score': f1
                }, model_path)
                print(f"Best model saved to: {model_path}")
        
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break
    
    return best_f1

# ---------- Main ----------
if __name__ == "__main__":
    # Paths & directories
    data_path = os.path.join('..', 'data', 'processed', 'track-b-clean.csv')
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Option 1: Run hyperparameter search (recommended first time)
    print("=== HYPERPARAMETER ===")
    best_config = hyperparameter_search(data_path, models_dir)
    
    # Option 2: Train with best configuration found
    print(f"\n=== TRAINING MODEL ===")
    best_config['epochs'] = 25  # Full training with more epochs
    final_f1 = train_with_config(data_path, best_config, models_dir, save_model=True)
    
    print(f"\nFinal F1 Score: {final_f1:.4f}")
  
    
  