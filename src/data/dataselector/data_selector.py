import pandas as pd

class DataFeatureSelector:
    @staticmethod
    def select_feature(dataset: pd.DataFrame, features: list) -> pd.DataFrame:
        dataset = dataset.copy()
        try:
            dataset = dataset.loc[:, features]
        except Exception as e:
            print(e)
        return dataset 