import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap.umap_ as umap
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

def analyze_biological_features(df):
    """
    Compare biological features between CpG Islands and non-CpG Islands
    """
    cpg_islands = df[df['CpG_island'] == 1]
    non_cpg_islands = df[df['CpG_island'] == 0]
    
    features_to_compare = ['perCpg', 'perGc', 'obsExp', 'seq_length', 'cpgNum', 'gcNum']
    
    results = []
    
    print("="*80)
    print("BIOLOGICAL FEATURE COMPARISON")
    print("="*80)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features_to_compare):
        cpg_vals = cpg_islands[feature].dropna()
        non_cpg_vals = non_cpg_islands[feature].dropna()
        
        # Statistical test
        from scipy.stats import mannwhitneyu
        stat, pval = mannwhitneyu(cpg_vals, non_cpg_vals)
        
        # Store results
        results.append({
            'feature': feature,
            'cpg_island_mean': cpg_vals.mean(),
            'cpg_island_std': cpg_vals.std(),
            'non_cpg_island_mean': non_cpg_vals.mean(),
            'non_cpg_island_std': non_cpg_vals.std(),
            'p_value': pval,
            'effect_size': (cpg_vals.mean() - non_cpg_vals.mean()) / np.sqrt((cpg_vals.std()**2 + non_cpg_vals.std()**2) / 2)
        })
        
        # Visualization
        ax = axes[idx]
        ax.hist(cpg_vals, bins=50, alpha=0.6, label='CpG Island', color='blue', density=True)
        ax.hist(non_cpg_vals, bins=50, alpha=0.6, label='Non-CpG Island', color='red', density=True)
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.set_title(f'{feature}\np-value: {pval:.2e}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('biological_features_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))
    print("\n" + "="*80)
    
    return results_df

def analyze_embedding_space(embeddings, df, n_samples=5000):
    """
    Analyze embedding space structure
    
    INPUT:
    - embeddings: numpy array of shape (n_samples, embedding_dim)
    - df: dataframe with metadata
    - n_samples: number of samples to visualize (for speed)
    
    OUTPUT:
    - Dimensionality reduction plots
    - Correlation analysis
    """
    embeddings_sample = embeddings
    df_sample = df.copy()
    
    print(f"\nAnalyzing {len(embeddings_sample)} samples")
    print(f"Embedding dimensionality: {embeddings_sample.shape[1]}")
    # ===================================================================
    # PART 2: Correlation between embedding dimensions and features

    # HYPOTHESIS: Embeddings should cluster by biological features, not spurious patterns
    # INPUT: Embeddings array and metadata
    # OUTPUT: Visualizations showing what drives embedding structure
    # ===================================================================

    print("\n2. Analyzing correlation between embeddings and biological features...")

    biological_features = ['perCpg', 'perGc', 'obsExp', 'seq_length', 'cpgNum', 'gcNum']

    # Calculate correlations for each embedding dimension
    n_dims = min(embeddings_sample.shape[1], 100)  # Analyze first 100 dimensions

    correlation_matrix = np.zeros((n_dims, len(biological_features)))

    for dim in range(n_dims):
        for feat_idx, feat in enumerate(biological_features):
            corr, _ = pearsonr(embeddings_sample[:, dim], df_sample[feat])
            correlation_matrix[dim, feat_idx] = corr

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(correlation_matrix, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)

    ax.set_xticks(range(len(biological_features)))
    ax.set_xticklabels(biological_features, rotation=45, ha='right')
    ax.set_ylabel('Embedding Dimension')
    ax.set_xlabel('Biological Feature')
    ax.set_title('Correlation: Embedding Dimensions vs Biological Features')

    plt.colorbar(im, ax=ax, label='Pearson Correlation')
    plt.tight_layout()
    plt.savefig('cpg_embedding_feature_correlations.png', dpi=300, bbox_inches='tight')

    print("   ✓ Saved: embedding_feature_correlations.png")

    # Find most correlated dimensions
    print("\n   Top embedding dimensions by feature correlation:")
    for feat_idx, feat in enumerate(biological_features):
        correlations = correlation_matrix[:, feat_idx]
        top_dim = np.argmax(np.abs(correlations))
        top_corr = correlations[top_dim]
        print(f"   {feat:12s}: Dimension {top_dim:3d} (r={top_corr:+.3f})")

    # ===================================================================
    # PART 3: PCA to identify main sources of variation
    # ===================================================================

    print("\n3. Running PCA to identify main variation sources...")

    from sklearn.decomposition import PCA
    pca = PCA(n_components=10)
    pca_coords = pca.fit_transform(embeddings_sample)

    # Explained variance
    print(f"\n   Explained variance by top 10 PCs: {pca.explained_variance_ratio_.sum():.2%}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # PC explained variance
    ax = axes[0, 0]
    ax.bar(range(1, 11), pca.explained_variance_ratio_)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('PCA Explained Variance')

    # PC1 vs PC2 colored by CpG Island status
    ax = axes[0, 1]
    for label in [0, 1]:
        mask = df_sample['CpG_island'] == label
        ax.scatter(pca_coords[mask, 0], pca_coords[mask, 1], 
                    alpha=0.5, s=10, label=f'CpG Island: {label}')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('PCA: Colored by CpG Island Status')
    ax.legend()

    # PC1 vs biological features
    ax = axes[1, 0]
    for feat in ['perCpg', 'obsExp']:
        scatter = ax.scatter(pca_coords[:, 0], df_sample[feat], 
                            alpha=0.3, s=10, label=feat)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel('Feature Value')
    ax.set_title('PC1 vs Biological Features')
    ax.legend()

    # Correlation of PCs with biological features
    ax = axes[1, 1]
    pc_feature_corr = np.zeros((10, len(biological_features)))
    for pc in range(10):
        for feat_idx, feat in enumerate(biological_features):
            corr, _ = pearsonr(pca_coords[:, pc], df_sample[feat])
            pc_feature_corr[pc, feat_idx] = corr

    im = ax.imshow(pc_feature_corr, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_yticks(range(10))
    ax.set_yticklabels([f'PC{i+1}' for i in range(10)])
    ax.set_xticks(range(len(biological_features)))
    ax.set_xticklabels(biological_features, rotation=45, ha='right')
    ax.set_title('PC Correlation with Features')
    plt.colorbar(im, ax=ax, label='Correlation')

    plt.tight_layout()
    plt.savefig('cpg_embedding_pca_analysis.png', dpi=300, bbox_inches='tight')

    print("   ✓ Saved: embedding_pca_analysis.png")

    # ===================================================================
    # PART 4: Check for spurious chromosomal patterns
    # ===================================================================

    print("\n4. Checking for chromosome-specific clustering (potential spurious signal)...")

    # Calculate silhouette score for chromosome clustering
    from sklearn.metrics import silhouette_score

    # Encode chromosomes numerically
    unique_chroms = df_sample['chrom'].unique()
    chrom_to_idx = {chrom: idx for idx, chrom in enumerate(unique_chroms)}
    chrom_labels = df_sample['chrom'].map(chrom_to_idx).values

    if len(unique_chroms) > 1:
        sil_score = silhouette_score(embeddings_sample, chrom_labels)
        print(f"   Silhouette score for chromosome clustering: {sil_score:.4f}")
        print(f"   (Lower is better - we DON'T want chromosome-specific patterns)")
        
        if sil_score > 0.1:
            print("   ⚠️  WARNING: Embeddings show chromosome-specific structure!")
            print("      This suggests the model may be learning genomic position rather than sequence features")
        else:
            print("   ✓ Good: No strong chromosome-specific clustering detected")

    # ===================================================================
    # PART 5: Within-class variation analysis
    # ===================================================================

    print("\n5. Analyzing within-class variation...")

    cpg_embeddings = embeddings_sample[df_sample['CpG_island'] == 1]
    non_cpg_embeddings = embeddings_sample[df_sample['CpG_island'] == 0]

    from sklearn.metrics import pairwise_distances

    cpg_distances = pairwise_distances(cpg_embeddings[:500], metric='euclidean')  # Sample for speed
    non_cpg_distances = pairwise_distances(non_cpg_embeddings[:500], metric='euclidean')

    # Get upper triangle (avoid diagonal and duplicates)
    cpg_dist_vals = cpg_distances[np.triu_indices_from(cpg_distances, k=1)]
    non_cpg_dist_vals = non_cpg_distances[np.triu_indices_from(non_cpg_distances, k=1)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(cpg_dist_vals, bins=50, alpha=0.6, label='CpG Island', density=True, color='blue')
    ax.hist(non_cpg_dist_vals, bins=50, alpha=0.6, label='Non-CpG Island', density=True, color='red')
    ax.set_xlabel('Euclidean Distance')
    ax.set_ylabel('Density')
    ax.set_title('Within-Class Distance Distribution')
    ax.legend()

    plt.tight_layout()
    plt.savefig('cpg_embedding_within_class_distances.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("   ✓ Saved: embedding_within_class_distances.png")
    print(f"\n   Mean within-class distance:")
    print(f"   CpG Island: {cpg_dist_vals.mean():.4f}")
    print(f"   Non-CpG Island: {non_cpg_dist_vals.mean():.4f}")

    # ===================================================================
    # SUMMARY
    # ===================================================================

    print("\n" + "="*80)
    print("INTERPRETATION GUIDE")
    print("="*80)
    print("\n✅ GOOD SIGNS (Model learning biological features):")
    print("   - UMAP clusters separate clearly by CpG_island label")
    print("   - Strong correlation between embedding dims and perCpg/obsExp")
    print("   - PC1/PC2 align with biological features")
    print("   - Low chromosome-specific clustering (silhouette < 0.1)")
    print("   - Similar within-class distances for both classes")

    print("\n⚠️  WARNING SIGNS (Potential spurious correlations):")
    print("   - Poor separation by CpG_island but strong by chromosome")
    print("   - High chromosome silhouette score (> 0.1)")
    print("   - Weak correlation with biological features")
    print("   - UMAP colored by seq_length shows clear clusters")
    print("   - Very different within-class distance distributions")

    print("\n" + "="*80)

    return {
        'pca_coords': pca_coords,
        'correlation_matrix': correlation_matrix,
        'pc_feature_corr': pc_feature_corr
    }


# Example usage:
df = pd.read_csv("/home/localuser/evo2/dmrs/classifier/bio_invest/AddCpgFeature.csv")
embeddings = np.load("/home/localuser/evo2/embeddings/evo2_1b_base_blocks_21_cpg_full_1dr_meanpool_fix.npy")
#results = analyze_embedding_space(embeddings, df)

# Run analysis
baseline_features = analyze_biological_features(df)