import numpy as np
import pandas as pd
import pyfaidx
import os
import subprocess

def download_genome():
    """Download mm39 genome if not exists"""
    if not os.path.exists('mm39.fa'):
        if not os.path.exists('mm39.fa.gz'):
            print("Downloading mm39 genome...")
            subprocess.run(['wget', 'https://hgdownload.soe.ucsc.edu/goldenPath/mm39/bigZips/mm39.fa.gz'])
        
        print("Extracting genome...")
        subprocess.run(['gunzip', 'mm39.fa.gz'])
    
    return pyfaidx.Fasta('mm39.fa')

def extract_sequence(row, genome):
    """Extract sequence from genome given coordinates"""
    try:
        chrom, start, end = row['chrom'], int(row['start']), int(row['end'])
        
        # Handle chromosome naming (ensure 'chr' prefix)
        if not str(chrom).startswith('chr'):
            chrom = f'chr{chrom}'
        
        # Extract sequence
        seq = genome[chrom][start:end].seq
        return str(seq).upper()
    
    except Exception as e:
        print(f"Error extracting sequence for {row}: {e}")
        return None

def add_sequences_to_dataframe(df, genome, sequence_col_name='sequence'):
    """Add sequence column to dataframe"""
    print(f"Extracting {len(df)} sequences...")
    df[sequence_col_name] = df.apply(lambda row: extract_sequence(row, genome), axis=1)
    
    # Remove rows where sequence extraction failed
    before_filter = len(df)
    df = df.dropna(subset=[sequence_col_name])
    after_filter = len(df)
    
    if before_filter != after_filter:
        print(f"Removed {before_filter - after_filter} rows due to failed sequence extraction")
    
    return df

def extract_by_feature(feature_name, dmrs_file, genome): 
    """Extract sequences for rows where feature == 1"""
    new_column_name = feature_name + "_sequence" 
    
    # Filter rows where feature == 1
    feature_rows = dmrs_file[dmrs_file[feature_name] == 1].copy()
    
    if len(feature_rows) > 0:
        feature_rows = add_sequences_to_dataframe(feature_rows, genome, new_column_name)
        return feature_rows
    else:
        print(f"No rows found with {feature_name} == 1")
        return pd.DataFrame()