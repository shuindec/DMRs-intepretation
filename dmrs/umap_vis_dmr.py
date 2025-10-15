import pandas as pd
import numpy as np
import seaborn as sns
import umap.umap_ as umap
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

dmrs = pd.read_csv("/home/localuser/evo2/dmrs/data/NewDmrs_1kbp_fixN.csv")
pooled_embeddings_np = np.load("/home/localuser/evo2/embeddings/evo2_1b_base_blocks_24_dmrs_237k_1dr_meanpool_fixN.npy")

reducer = umap.UMAP(n_components=2, random_state=42)
embedding_2d = reducer.fit_transform(pooled_embeddings_np)
# try tunning with: umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)

dmrs['diff'] = round(dmrs['mean_value_mut'] - dmrs['mean_value_wt'], 2)
dmrs['class'] = 0 # is 'not-DMR'
dmrs.loc[(dmrs['diff'] > 0) & (dmrs['is_dmr'] == True), 'class'] = 1 #'hyper'
dmrs.loc[(dmrs['diff'] < 0) & (dmrs['is_dmr'] == True), 'class'] = 2 #'hypo'

# Create DataFrame for plotting
umap_df = pd.DataFrame(embedding_2d, columns=['UMAP1', 'UMAP2'])
umap_df['class'] = dmrs['is_dmr'].apply(lambda x: 0 if 'True' in str(x) else 1).values.astype(np.float32)
umap_df['level'] = dmrs['class'].values #nonDMR, hyper or hypo
#umap_df['class'] = dmrs['CpG_island'].values #target
# umap_df['CpG_status'] = dmrs['cpg_island_type'].values #categorical
# umap_df['cpgNum'] = dmrs['cpgNum'].values #continous
# umap_df['seq_length'] = dmrs['seq_length'].values #continous
# umap_df['perGc'] = dmrs['perGc'].values #continous
#df['label_encoded'] = df['label'].map({'DMR': 1, 'non-DMR': 0}) #encoder class

sil_score = silhouette_score(embedding_2d, umap_df['class']) # If class balance is highly skewed, silhouette might be misleading.

## PLot separately

plt.figure(figsize=(15, 5))

# Plot 1: 
plt.subplot(1, 3, 1)
scatter = plt.scatter(
    umap_df['UMAP1'],
    umap_df['UMAP2'],
    c=umap_df['class'],          # continuous variable for color
    cmap='viridis',               # or 'plasma', 'coolwarm',....
    alpha=0.7
)
plt.title('By 2 Classes')

# Plot 2: 
plt.subplot(1, 3, 2)
#class_colors = pd.Categorical(df['class']).codes
scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=umap_df['level'], alpha=0.6, s=20, cmap='viridis')
plt.title('By 3 Classes')
plt.colorbar(scatter)
plt.tight_layout()

# Plot 3: 
# plt.subplot(1, 3, 3)
# #class_colors = pd.Categorical(df['class']).codes
# scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=umap_df['seq_length'], alpha=0.6, s=20, cmap='viridis')
# plt.title('By seq_length Class')
# plt.colorbar(scatter)
# plt.tight_layout()


plt.savefig("umapDmr.png")
#plt.show()