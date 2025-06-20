# src/data_ingest.py

import logging
from pathlib import Path
import pandas as pd


class DataIngestor:
    """
    DataIngestor handles reading raw CSV price files, aligning them on a common calendar,
    filling small gaps, and saving the cleaned, merged DataFrame to disk.
    """

    def __init__(self,
                 assets: dict,
                 start_date: str,
                 end_date: str,
                 raw_dir: Path = None,
                 processed_file: Path = None):
        """
        Initialize the DataIngestor.

        :param assets: dict mapping asset names to tickers (keys: names, values: tickers)
        :param start_date: inclusive start date in 'YYYY-MM-DD' format
        :param end_date: inclusive end date in 'YYYY-MM-DD' format
        :param raw_dir: Path to directory containing raw CSVs (default: project/data/raw)
        :param processed_file: Path for saving the cleaned CSV
                               (default: project/data/processed/prices.csv)
        """
        self.assets = assets
        self.start = pd.to_datetime(start_date)
        self.end = pd.to_datetime(end_date)
        # Default paths relative to this file’s parent directory
        base = Path(__file__).parent.parent
        self.raw_dir = raw_dir or base / "data" / "raw"
        self.processed_file = processed_file or base / "data" / "processed" / "prices.csv"

        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_file.parent.mkdir(parents=True, exist_ok=True)

        # Logging setup
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(message)s")
        self.logger = logging.getLogger(self.__class__.__name__)

        # Placeholder for loaded DataFrames
        self.data_frames = {}

    def load_raw(self) -> None:
        """
        Load each asset's raw CSV into a pandas DataFrame, parsing dates, sorting,
        and storing in self.data_frames[name] = DataFrame(name=Close).
        """
        for name in self.assets:
            path = self.raw_dir / f"{name}.csv"
            if not path.exists():
                self.logger.error(f"Missing raw file: {path}")
                raise FileNotFoundError(f"{path} not found")

            df = pd.read_csv(path, parse_dates=True, index_col=0)
            df = df.sort_index()
            if "Close" not in df.columns:
                self.logger.error(f"No 'Close' column in {path.name}")
                raise KeyError(f"No 'Close' in {path.name}")

            # Keep only the Close price, rename column to asset name
            self.data_frames[name] = df[["Close"]].rename(columns={"Close": name})
            self.logger.info(f"Loaded {name}: {len(df)} rows from {path.name}")

    def align_and_clean(self) -> pd.DataFrame:
        """
        Align all asset DataFrames on a business-day index, fill single-day gaps,
        drop any remaining NaNs, and return the merged DataFrame.
        """
        # Create Mon–Fri index between start and end
        bidx = pd.date_range(self.start, self.end, freq="B")
        merged = pd.DataFrame(index=bidx)

        # Align each asset, fill small gaps
        for name, df in self.data_frames.items():
            aligned = df.reindex(bidx)
            # Forward-fill then back-fill at most 1 day
            aligned[name].ffill(limit=1, inplace=True)
            aligned[name].bfill(limit=1, inplace=True)
            merged[name] = aligned[name]
            self.logger.info(f"{name}: {merged[name].isna().sum()} missing after fill")

        # Drop any rows still containing NaNs
        before = len(merged)
        merged.dropna(inplace=True)
        dropped = before - len(merged)
        if dropped:
            self.logger.warning(f"Dropped {dropped} rows with NaNs")

        self.logger.info(f"Merged data has {merged.shape[0]} rows × {merged.shape[1]} assets")
        return merged

    def save_processed(self, df: pd.DataFrame) -> None:
        """
        Save the cleaned DataFrame to CSV at self.processed_file.
        """
        df.to_csv(self.processed_file)
        self.logger.info(f"Saved cleaned prices to {self.processed_file}")

    def run(self) -> None:
        """
        Execute the full pipeline: load raw data, align & clean, then save.
        """
        self.load_raw()
        clean_df = self.align_and_clean()
        self.save_processed(clean_df)


if __name__ == "__main__":
    # Example invocation
    assets = {"SPX": "^GSPC", "GOLD": "GC=F", "BTC": "BTC-USD"}
    ingestor = DataIngestor(
        assets=assets,
        start_date = "2022-01-01",
        end_date = "2025-01-01"
    )
    ingestor.run()