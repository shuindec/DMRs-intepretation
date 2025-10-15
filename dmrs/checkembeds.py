import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, matthews_corrcoef, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

df = pd.read_csv("/home/localuser/evo2/dmrs/data/NewDmrs_1kbp_fixN.csv")
embeddings = np.load("/home/localuser/evo2/embeddings/evo2_1b_base_blocks_24_dmrs_237k_1dr_meanpool_fixN.npy")

labels = df['is_dmr'].apply(lambda x: 0 if 'True' in str(x) else 1)
#labels = df['is_dmr'].values.astype(np.float32)

print(f"Embeddings shape: {embeddings.shape}")  # (num_samples, embedding_dim)
print(f"Labels shape: {labels.shape}")  # (num_samples,)
print(f"DMR samples: {(labels == 1).sum().item()}")
print(f"Non-DMR samples: {(labels == 0).sum().item()}")

# Convert to numpy for analysis
embeddings_np = embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings

scaler = StandardScaler()
embeddings_np = scaler.fit_transform(embeddings_np)
# Basic statistics
print(f"Embedding shape: {embeddings_np.shape}")
print(f"Data type: {embeddings_np.dtype}")
print(f"\nStatistics:")
print(f"  Mean: {embeddings_np.mean():.6f}")
print(f"  Std: {embeddings_np.std():.6f}")
print(f"  Min: {embeddings_np.min():.6f}")
print(f"  Max: {embeddings_np.max():.6f}")
print(f"  Median: {np.median(embeddings_np):.6f}")

# Check for problems
num_zeros = (embeddings_np == 0).sum()
total_elements = embeddings_np.size
pct_zeros = (num_zeros / total_elements) * 100

print(f"\n🚨 Quality Checks:")
print(f"  Zero values: {num_zeros:,} / {total_elements:,} ({pct_zeros:.2f}%)")
print(f"  NaN values: {np.isnan(embeddings_np).sum()}")
print(f"  Inf values: {np.isinf(embeddings_np).sum()}")
print(f"  Unique values: {len(np.unique(embeddings_np)):,}")


## Check if embeddings being classified properly with simplest model
X_train, X_test, y_train, y_test = train_test_split(
        embeddings_np, labels, test_size=0.2, random_state=42 )

sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
    
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_bal, y_train_bal)
acc = accuracy_score(y_test, clf.predict(X_test))
auc = roc_auc_score(y_test, clf.predict(X_test))
cm = confusion_matrix(y_test, clf.predict(X_test))
print(cm)

recall = recall_score(y_test, clf.predict(X_test))
print(f"Recall: {recall:.3f}")

mcc = matthews_corrcoef(y_test, clf.predict(X_test))
print(f"MCC: {mcc:.3f}")
print(f"🎯 Logistic Regression AUC-ROC: {auc:.2%}")    
print(f"🎯 Logistic Regression Accuracy: {acc:.2%}")
print(f"   > 70%: ✅ Good embeddings")
print(f"   < 60%: ❌ Bad embeddings")


### Check feature important

vals = embeddings_np[:, 1202]
print("Mean:", np.mean(vals))
print("Std:", np.std(vals))
print("Min:", np.min(vals))
print("Max:", np.max(vals))
print("Unique values:", np.unique(vals))

import matplotlib.pyplot as plt
plt.hist(vals, bins=50)
plt.title("Distribution of Embedding Dimension 1202")
plt.savefig("1202_featureCheck.png")

import seaborn as sns
sns.boxplot(x=df['is_dmr'], y=vals)
plt.xlabel('Label')
plt.ylabel('Embedding Value (1202)')
plt.savefig("1202_featureCheck2.png")

from scipy.stats import ttest_ind
vals_0 = vals[df['is_dmr']==0]
vals_1 = vals[df['is_dmr']==1]
print("T-test: If the separation is extreme or the value is 'one-hot' with the label, check for data leakage.")
print(ttest_ind(vals_0, vals_1))

#Check for Correlations
corr = np.corrcoef(vals, df['is_dmr'])[0,1]
print("Correlation with label:", corr)