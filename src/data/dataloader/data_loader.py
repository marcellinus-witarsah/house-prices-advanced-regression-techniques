import os
import pandas as pd
from typing import Optional

class DataLoader:
    @staticmethod
    def load_data(path: str) -> Optional[pd.DataFrame]:
        if os.path.exists(path):
            return pd.read_csv(path)
        return None