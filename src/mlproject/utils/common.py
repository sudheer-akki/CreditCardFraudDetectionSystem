import numpy as np

def introduce_missing_values(df, fraction=0.1):
    df_raw = df.copy()
    total_values = df_raw.size
    num_missing = int(total_values * fraction)

    # Randomly select indices to replace with NaN
    for _ in range(num_missing):
        row_idx = np.random.randint(0, df_raw.shape[0])
        col_idx = np.random.randint(0, df_raw.shape[1])
        df_raw.iloc[row_idx, col_idx] = np.nan
    return df_raw