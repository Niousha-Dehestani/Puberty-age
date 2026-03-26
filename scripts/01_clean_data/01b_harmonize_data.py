"""
Script: 01b_harmonize_data.py
Description: Harmonizes ABCD PDS and Hormone data across collection sites
             using neuroCombat. Data is split by sex to prevent NaN conflicts
             between male/female specific PDS items, then recombined.
"""

import pandas as pd
import numpy as np
from neuroCombat import neuroCombat
from pathlib import Path

# --- PATHS (Your Local Mac Path) ---
DATA_DIR = Path('/Users/dir/Documents/abcd6.0/processed')
INPUT_FILE = DATA_DIR / '01_final_cleaned_data_with_hormones.csv'
OUTPUT_FILE = DATA_DIR / '01b_harmonized_data.csv'

def run_combat_for_group(df_group, features, batch_col, categorical_covars, continuous_covars):
    """Helper function to run ComBat on a specific sex group."""
    subset_cols = [batch_col] + categorical_covars + continuous_covars + features
    df_clean = df_group.dropna(subset=subset_cols).copy()
    
    if len(df_clean) < 2:
        return df_clean # Not enough data to harmonize

    # Prepare matrices
    data_matrix = df_clean[features].T.values
    covars = df_clean[[batch_col] + categorical_covars + continuous_covars].copy()

    # Run neuroCombat
    combat_output = neuroCombat(
        dat=data_matrix,
        covars=covars,
        batch_col=batch_col,
        categorical_cols=categorical_covars,
        continuous_cols=continuous_covars
    )

    # Put harmonized data back
    df_clean[features] = combat_output['data'].T 
    return df_clean

def run_harmonization():
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)

    # 1. REMOVE SITE 22
    initial_len = len(df)
    df = df[df['ab_g_dyn__design_site'] != 22].copy()
    print(f"Removed Site 22: Dropped {initial_len - len(df)} rows.")

    # 2. DEFINE COLUMNS
    batch_col = 'ab_g_dyn__design_site'
    # Since we split by sex, sex is no longer a covariate for ComBat!
    categorical_covars = [] 
    continuous_covars = ['ab_g_dyn__visit_age']

    # Define features for FEMALES (Sex == 2)
    female_features = [
        'tst_filter_rsi_hormone', 'dhea_filter_rsi_hormone',
        'ph_p_pds_001', 'ph_p_pds_002', 'ph_p_pds_003',
        'ph_p_pds__f_001', 'ph_p_pds__f_002'
    ]

    # Define features for MALES (Sex == 1)
    male_features = [
        'tst_filter_rsi_hormone', 'dhea_filter_rsi_hormone',
        'ph_p_pds_001', 'ph_p_pds_002', 'ph_p_pds_003',
        'ph_p_pds__m_001', 'ph_p_pds__m_002'
    ]

    # 3. SPLIT DATA BY SEX
    # NOTE: Assuming column is 'sex' and coding is 1=Male, 2=Female
    df_female = df[df['sex'] == 2].copy()
    df_male = df[df['sex'] == 1].copy()

    print(f"Starting sample sizes - Females: {len(df_female)}, Males: {len(df_male)}")

    # 4. RUN HARMONIZATION SEPARATELY
    print("\nHarmonizing Female data...")
    df_female_harmonized = run_combat_for_group(df_female, female_features, batch_col, categorical_covars, continuous_covars)
    
    print("Harmonizing Male data...")
    df_male_harmonized = run_combat_for_group(df_male, male_features, batch_col, categorical_covars, continuous_covars)

    print(f"\nFinal harmonized sizes - Females: {len(df_female_harmonized)}, Males: {len(df_male_harmonized)}")

    # 5. RECOMBINE AND SAVE
    final_df = pd.concat([df_female_harmonized, df_male_harmonized], ignore_index=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Harmonized data successfully merged and saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_harmonization()
