#!/usr/bin/env python3
"""
Quick test script for Evo2 embedding pipeline
Run this first to debug issues with a small subset
"""

import pandas as pd
import torch
from models import Evo2EmbedderSimple
import numpy as np
from tqdm import tqdm

def reverse_sequence(sequence):
    complement = ''
    sequence = sequence.upper()
    for x in sequence:
        if x == 'A':
            y = 'T'
        elif x == 'T':
            y = 'A'
        elif x == 'G':
            y = 'C'
        elif x == 'C':
            y = 'G'
        else:
            y = 'N'
        complement += y
    # Return the reverse of the complement
    return complement[::-1]

def create_subset_for_layer_testing(df):
    df_dmr = df[df['CpG_island'] == 1].sample(n=50) #randomly select row
    df_non_dmr = df[df['CpG_island'] == 0].sample(n=50)
    df_sub = pd.concat([df_dmr, df_non_dmr])

    # Add the reverse_seq column
    df_sub['rev_seq'] = df_sub['sequence'].apply(reverse_sequence)
    return df_sub

def create_dmrs_with_reverse_seq(df):
    df['rev_seq'] = df['sequence'].apply(reverse_sequence)
    df.to_csv('NewDmrs_1kbp_fixN_2dr.csv', index=False)
    return df

def quick_test():
    
    ## create and save test subset
    dmrs = pd.read_csv("/home/localuser/evo2/dmrs/data/NewDmrs_1kbp_fixN.csv")
    df = create_dmrs_with_reverse_seq(dmrs)
    
    
    print("Loading sequences...")
    ## Generate embeddings for CpG Islands
    # df = pd.read_csv("/home/localuser/evo2/dmrs/data/NewDmrs_1kbp_fixN.csv")
    # #df.to_csv("test_subset_3k", index=False,)
    # df = df.iloc[df['sequence'].str.len().argsort()] # fix very long sequences 6600 bp at location 17298
    

    #should remove chrX, chrY and chrUn_GL456370v1
    layers = ['blocks.23'] #'blocks.29.mlp.l3', 'blocks.30.mlp.l3', 'blocks.31.mlp.l3'
    
    embedder = Evo2EmbedderSimple('evo2_1b_base')
    
    for layer in layers:
        embedding = embedder.embed_sequence(df, layer)

    print(f"\n Batch processing successful!")
    print(f"Final batch shape: {embedding[1].shape}")
    print(f"Batch mean: {embedding[1].mean():.4f}")
    #print(f"Zero embeddings: {np.all(batch_embeddings == 0, axis=1).sum()}")
    
    return True

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n✅ Ready to run full pipeline!")
    else:
        print("\n❌ Fix the issues above before running the full pipeline.")

        