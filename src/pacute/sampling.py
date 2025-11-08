import pandas as pd
from pathlib import Path

def load_frequency_data(freq_file_path='data/word_frequencies.csv'):
    return pd.read_csv(freq_file_path)

def add_frequency_ranks(df, freq_df, word_column='normalized_word'):
    df_copy = df.copy()
    df_copy['normalized_lower'] = df_copy[word_column].str.lower()

    result = df_copy.merge(
        freq_df,
        left_on='normalized_lower',
        right_on='normalized',
        how='left'
    )

    result['rank'] = result['rank'].fillna(100000)
    result = result.drop(columns=['normalized_lower', 'normalized'])

    return result

def sample_by_frequency(df, n_samples, freq_weight=0.5, random_state=42):
    if 'rank' not in df.columns:
        raise ValueError("DataFrame must have 'rank' column. Use add_frequency_ranks() first.")

    df = df.copy()

    df['sample_weight'] = 1.0 / (df['rank'] + 1)

    uniform_weight = 1.0 / len(df)
    df['sample_weight'] = (
        freq_weight * df['sample_weight'] +
        (1 - freq_weight) * uniform_weight
    )

    df['sample_weight'] = df['sample_weight'] / df['sample_weight'].sum()

    return df.sample(
        n=min(n_samples, len(df)),
        weights='sample_weight',
        random_state=random_state,
        replace=False
    ).drop(columns=['sample_weight'])
