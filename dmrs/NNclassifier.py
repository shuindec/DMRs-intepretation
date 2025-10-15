import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, matthews_corrcoef, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class EmbeddingDataset(Dataset):
    """Custom dataset for pre-extracted embeddings."""
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class CpGClassifier(nn.Module):
    def __init__(self, input_dim=1920, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
            #nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x).squeeze(1)

class CpGClassifierV2(nn.Module):
    def __init__(self, input_dim=1920, hidden_dims=[512, 256, 128, 64]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  # Add batch normalization
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(1)
df = pd.read_csv("/home/localuser/evo2/dmrs/data/NewDmrs_1kbp_fixN.csv")
embeddings = np.load("/home/localuser/evo2/embeddings/evo2_1b_base_blocks_24_dmrs_237k_1dr_meanpool_fixN.npy")

labels = df['is_dmr'].apply(lambda x: 0 if 'True' in str(x) else 1).values.astype(np.float32)
#labels = df['CpG_island'].values.astype(np.float32)  # shape (N,)

## Check input
# print("Embeddings shape:", embeddings.shape)  # (5571, 1920)
# print("Labels shape:", labels.shape)          # (5571,)

# print("Label counts:", pd.Series(labels).value_counts()) #0: 3255, 1:2316
# print(np.isnan(embeddings).any(), np.isinf(embeddings).any()) #False False

scaler = StandardScaler()
X_scaled = scaler.fit_transform(embeddings)
print(f"After scaling - Mean: {X_scaled.mean():.4f}, Std: {X_scaled.std():.4f}")

# 1st split: 80% train+val, 20% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_scaled, labels, test_size=0.2, random_state=42, stratify=labels )

# Second split: 80% train, 20% val (of the train+val set)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val)

# Create datasets
train_dataset = EmbeddingDataset(X_train, y_train)
val_dataset = EmbeddingDataset(X_val, y_val)
test_dataset = EmbeddingDataset(X_test, y_test)

print(f"Train - Validation - Test Split: {len(train_dataset)}, {len(val_dataset)}, {len(test_dataset)}")

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


#handle classimbalance
# class_weights = total / (num_classes * class_counts)
n_neg = (y_train == 0).sum()  # Non-DMR/ non-CpG count
n_pos = (y_train == 1).sum()  # DMR/CpG count
# Use weighted loss
pos_weight = torch.tensor([n_neg / n_pos]).to(device)

print(f"Using pos_weight: {pos_weight.item():.3f}")

# Model, optimizer, loss
model = CpGClassifierV2(input_dim=1920).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # Combines sigmoid + BCE for numerical stability

# Training loop
num_epochs = 50
best_val_loss = float('inf')
patience = 10
patience_counter = 0

print("\nStarting training...")
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    train_preds = []
    train_labels = []
    
    for batch_embeddings, batch_labels in train_loader:
        batch_embeddings = batch_embeddings.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        logits = model(batch_embeddings)
        loss = loss_fn(logits, batch_labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_preds.extend(torch.sigmoid(logits).cpu().detach().numpy())
        train_labels.extend(batch_labels.cpu().numpy())
    
    train_loss /= len(train_loader)
    train_acc = accuracy_score(train_labels, np.array(train_preds) >= 0.5)
    
    # Validation
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for batch_embeddings, batch_labels in val_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)
            
            logits = model(batch_embeddings)
            loss = loss_fn(logits, batch_labels)
            
            val_loss += loss.item()
            val_preds.extend(torch.sigmoid(logits).cpu().numpy())
            val_labels.extend(batch_labels.cpu().numpy())
    
    val_loss /= len(val_loader)
    val_acc = accuracy_score(val_labels, np.array(val_preds) >= 0.5)
    val_auroc = roc_auc_score(val_labels, val_preds)
    
    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUROC: {val_auroc:.4f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        #torch.save(model.state_dict(), "best_cpg_classifier.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break


# Final evaluation on test set
print("\n" + "="*50)
print("FINAL EVALUATION ON TEST SET")
print("="*50)

model.eval()
test_preds = []
test_labels_list = []

with torch.no_grad():
    for batch_embeddings, batch_labels in test_loader:
        batch_embeddings = batch_embeddings.to(device)
        logits = model(batch_embeddings)
        probs = torch.sigmoid(logits).cpu().numpy()
        test_preds.extend(probs)
        test_labels_list.extend(batch_labels.cpu().numpy())

test_preds = np.array(test_preds)
test_labels_array = np.array(test_labels_list).flatten()
test_preds_binary = (test_preds >= 0.5).astype(int).flatten()

test_auroc = roc_auc_score(test_labels_array, test_preds)
test_acc = accuracy_score(test_labels_array, test_preds_binary)

print(f"\n🎯 Test AUROC: {test_auroc:.4f}")
print(f"🎯 Test Accuracy: {test_acc:.4f}")
print(f"\n📊 Classification Report:")
print(classification_report(test_labels_array, test_preds_binary, 
                          target_names=['Non-CpG', 'CpG']))

cm = confusion_matrix(test_labels_array, test_preds_binary)
print(f"\nConfusion Matrix:\n{cm}")

# Compute MCC
mcc = matthews_corrcoef(test_labels_array, test_preds_binary)
print(f"MCC: {mcc:.4f}")