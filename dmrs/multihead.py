import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                             matthews_corrcoef, recall_score, roc_curve, precision_recall_curve, average_precision_score)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Data loading
df = pd.read_csv("/home/localuser/evo2/dmrs/data/cpg_full_sorted.csv")
embeddings = np.load("/home/localuser/evo2/embeddings/evo2_1b_base_blocks_21_cpg_full_1dr_meanpool_fix.npy")
dmr_labels = df['CpG_island']

if dmr_labels.sum() > 0 and (dmr_labels == 0).sum() > 0:
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, dmr_labels, test_size=0.2, random_state=42, stratify=dmr_labels
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train_scaled, y_train)

    # Initialize models
    models = {
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train_bal, y_train_bal)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        cm = confusion_matrix(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        recall = recall_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'cm': cm,
            'auc': auc_score,
            'fpr': fpr,
            'tpr': tpr,
            'recall': recall,
            'mcc': mcc
        }
        print(f"\n{name} Results:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", cm)
        print(f"Recall: {recall:.3f} | MCC: {mcc:.3f} | AUC-ROC: {auc_score:.3f}")

    # ROC Curve
    plt.figure(figsize=(9, 7))
    for name, res in results.items():
        plt.plot(res['fpr'], res['tpr'], label=f"{name} (AUC={res['auc']:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Multiple Models: XGBoost, LogisticRegression, RandomForest)")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("roc_curve_all_models.png")
    plt.show()

    # Confusion Matrix Heatmaps
    for name, res in results.items():
        plt.figure(figsize=(5, 4))
        sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues')
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f"conf_matrix_{name}.png")
        plt.show()