from transformers import AutoTokenizer
import torch
from torch.utils.data import TensorDataset
import re
import pandas as pd
import os
from data_prep import clean_text
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import random_split, DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score
from train import calculate_multilabel_f1, eval_epoch, train_epoch

def word_tokenizer(texts):
    """
    Tokenises the input text into a list of words.
    """
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    encodings = tokenizer(
        texts,
        padding="max_length",    
        truncation=True,         
        max_length=128,          
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

def prepare_dataset(data_path):
    """
    Prepare dataset for multi-label emotion intensity classification.
    Each emotion has intensity values from 0-3.
    """
    clean_data = pd.read_csv(data_path, encoding='unicode_escape')
    texts = clean_data["text"].tolist()
    encoding = word_tokenizer(texts)
    
    # Convert emotion intensity labels (0-3) to long tensor for CrossEntropyLoss
    emotion_cols = ["anger", "fear", "joy", "sadness", "surprise"]
    labels = torch.tensor(clean_data[emotion_cols].values, dtype=torch.long)
    
    # Verify label ranges are correct (0-3)
    print(f"Label ranges: min={labels.min().item()}, max={labels.max().item()}")
    assert labels.min() >= 0 and labels.max() <= 3, "Labels must be in range 0-3"
    
    tensor_dataset_obj = create_tensor_dataset(encoding, labels)
    print("TensorDataset created with {} samples.".format(len(tensor_dataset_obj)))
    return tensor_dataset_obj
  
def load_data(data_path, batch_size=16):
    tensorDataset = prepare_dataset(data_path)
    total_size = len(tensorDataset)
    val_size = int(0.1 * total_size)       
    train_size = total_size - val_size

    train_ds, val_ds = random_split(tensorDataset, [train_size, val_size])   

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size) 
    return train_loader, val_loader

class IntensityClassifier(nn.Module):
    def __init__(self, pretrained_model="google-bert/bert-base-uncased", num_classes=4):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(pretrained_model)
        hidden = self.backbone.config.hidden_size
        self.emotion_names = ["anger", "fear", "joy", "sadness", "surprise"]
        
        # Create separate classification heads for each emotion
        self.heads = nn.ModuleDict({
            emo: nn.Linear(hidden, num_classes)
            for emo in self.emotion_names
        })

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Get logits for each emotion
        logits = {emo: head(cls) for emo, head in self.heads.items()}
        return logits

# Example train_epoch function (if you need it)
def train_epoch_example(model, train_loader, optimizer, criterion, device):
    """
    Training epoch for multi-label emotion intensity classification.
    Each emotion is classified independently into 4 intensity levels (0-3).
    """
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        
        # Calculate loss for each emotion separately
        # labels shape: [batch_size, 5] where each column is emotion intensity (0-3)
        loss = 0
        for i, emo in enumerate(model.emotion_names):
            emo_labels = labels[:, i]  # Shape: [batch_size] with values 0-3
            emo_logits = logits[emo]   # Shape: [batch_size, 4] (4 intensity classes)
            loss += criterion(emo_logits, emo_labels)
        
        # Average the loss across emotions
        loss = loss / len(model.emotion_names)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

# Example eval_epoch function (if you need it)
def eval_epoch_example(model, val_loader, device):
    """
    Evaluation epoch for multi-label emotion intensity classification.
    Returns predictions and true labels for each emotion.
    """
    model.eval()
    all_preds = {emo: [] for emo in model.emotion_names}
    all_labels = {emo: [] for emo in model.emotion_names}
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            logits = model(input_ids, attention_mask)
            
            for i, emo in enumerate(model.emotion_names):
                # Get predicted intensity class (0-3)
                preds = torch.argmax(logits[emo], dim=1)
                all_preds[emo].extend(preds.cpu().numpy())
                all_labels[emo].extend(labels[:, i].cpu().numpy())
    
    return all_preds, all_labels

# Example F1 calculation function (if you need it)
def calculate_multilabel_f1_example(all_labels, all_preds):
    """
    Calculate macro F1 score across all emotions.
    For intensity classification, we can treat it as multi-class classification for each emotion.
    """
    f1_scores = []
    
    for emo in ["anger", "fear", "joy", "sadness", "surprise"]:
        f1 = f1_score(all_labels[emo], all_preds[emo], average='macro')
        f1_scores.append(f1)
        print(f"{emo.capitalize()} F1: {f1:.4f}")
    
    macro_f1 = sum(f1_scores) / len(f1_scores)
    print(f"Overall Macro F1: {macro_f1:.4f}")
    return macro_f1

# ---------- Main ----------
if __name__ == "__main__":
    # Paths & hyperparams
    data_path = os.path.join('..', 'data', 'processed', 'track-b-clean.csv')
    epochs = 14
    lr = 2e-5
    batch_size = 16
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Load data
    train_loader, val_loader = load_data(data_path, batch_size=batch_size)

    # Device & model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = IntensityClassifier().to(device)

    # Optimizer & loss
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0

    # Training loop
    for epoch in range(1, epochs + 1):
        # Use your existing functions or the example ones above
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        preds, labels = eval_epoch(model, val_loader, device)
        
        f1 = calculate_multilabel_f1(labels, preds)
        print(f"Epoch {epoch}: TrainLoss={train_loss:.4f}, ValMacroF1={f1:.4f}")

        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            model_path = os.path.join(models_dir, 'best_model.pt')
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved to: {model_path}")