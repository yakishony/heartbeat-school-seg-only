import pandas as pd


def dataset_to_summary_df(dataset: dict) -> pd.DataFrame:
    """One-row-per-recording summary with rec_id, type, sr, n_samples, length_sec."""
    return pd.DataFrame([
        {'rec_id': rec_id, 'type': rec['type'], 'sr': rec['sr'],
         'n_samples': len(rec['signal']), 'length_sec': len(rec['signal']) / rec['sr']}
        for rec_id, rec in dataset.items()
    ])
