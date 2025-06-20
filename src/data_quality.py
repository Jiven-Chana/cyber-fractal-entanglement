# src/data_quality.py

import logging
from pathlib import Path
from typing import Union, Dict
import pandas as pd



class DataQualityChecker:
    """
    Data quality assessment for a price DataFrame.

    Methods:
      - load_data(): load CSV into DataFrame
      - check_missing(): count NaNs
      - check_duplicates(): count duplicate dates
      - check_frequency(): infer index frequency
      - head_tail(): show top & bottom rows
      - compute_return_stats(): volatility, skew, kurtosis
      - correlation_matrix(): return correlation DataFrame
      - run_all(): run all checks and return results dict
      - save_report(): flatten and save all results to CSV
    """

    def __init__(self,
                 filepath: Union[str, Path],
                 date_column: str = "Date",
                 index_col: str = "Date",
                 parse_dates: bool = True):
        self.filepath = Path(filepath)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(handler)

        # Load the data
        df = pd.read_csv(self.filepath, parse_dates=[date_column] if parse_dates else None)
        self.logger.info(f"Loaded {len(df)} rows from {self.filepath}")
        if index_col in df.columns:
            df = df.set_index(index_col)
        self.df = df

    def check_missing(self) -> pd.Series:
        missing = self.df.isna().sum()
        self.logger.info(f"Missing values per column:\n{missing}")
        return missing

    def check_duplicates(self) -> int:
        dup_count = int(self.df.index.duplicated().sum())
        self.logger.info(f"Duplicate index entries: {dup_count}")
        return dup_count

    def check_frequency(self) -> str:
        freq = self.df.index.inferred_freq or "Unknown"
        self.logger.info(f"Inferred index frequency: {freq}")
        return freq

    def head_tail(self, n: int = 5) -> pd.DataFrame:
        ht = pd.concat([self.df.head(n), self.df.tail(n)])
        self.logger.info(f"Head and tail (n={n}):\n{ht}")
        return ht

    def compute_return_stats(self) -> pd.DataFrame:
        ret = self.df.pct_change().dropna()
        stats = pd.DataFrame({
            'volatility': ret.std(),
            'skew': ret.skew(),
            'kurtosis': ret.kurtosis()
        })
        self.logger.info(f"Return statistics:\n{stats}")
        return stats

    def correlation_matrix(self) -> pd.DataFrame:
        ret = self.df.pct_change().dropna()
        corr = ret.corr()
        self.logger.info(f"Correlation matrix:\n{corr}")
        return corr

    def run_all(self) -> Dict[str, Union[pd.Series, pd.DataFrame, int, str]]:
        results = {
            'missing': self.check_missing(),
            'duplicates': self.check_duplicates(),
            'frequency': self.check_frequency(),
            'head_tail': self.head_tail(),
            'return_stats': self.compute_return_stats(),
            'correlation_matrix': self.correlation_matrix()
        }
        return results

    def save_report(self, report_path: Union[str, Path]) -> None:
        rp = Path(report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        results = self.run_all()
        rows = []
        for key, val in results.items():
            if isinstance(val, pd.DataFrame):
                df_flat = val.copy()
                df_flat['metric'] = key
                # Reset index, capturing its column name automatically
                df_reset = df_flat.reset_index()
                index_col = df_reset.columns[0]
                # Melt all other columns (except metric and the index column)
                rows.append(
                    df_reset.melt(
                        id_vars=[index_col, 'metric'],
                        var_name='asset',
                        value_name='value'
                    )
                )
            elif isinstance(val, pd.Series):
                df_flat = val.rename_axis('asset').reset_index(name='value')
                df_flat['metric'] = key
                rows.append(df_flat)
            else:
                rows.append(pd.DataFrame({'metric': [key], 'value': [val]}))

        report_df = pd.concat(rows, ignore_index=True)
        report_df.to_csv(rp, index=False)
        self.logger.info(f"Saved data quality report to {rp}")


if __name__ == "__main__":
    # import argparse
    #
    # parser = argparse.ArgumentParser(description="Run data quality checks.")
    # parser.add_argument("input_csv", help="Processed prices CSV")
    # parser.add_argument("--output", "-o",
    #                     default="reports/data_quality_report.csv",
    #                     help="Where to save the report")
    # args = parser.parse_args()
    #
    # dq = DataQualityChecker(args.input_csv)
    # dq.save_report(args.output)
    import os

    print("Working directory is:", os.getcwd())
    print("Looking for file at:", Path("data/processed/prices.csv").resolve())
    dq = DataQualityChecker("data/processed/prices.csv")
    dq.save_report("reports/data_quality_report.csv")