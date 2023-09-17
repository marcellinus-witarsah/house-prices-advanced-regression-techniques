import pandas as pd

class DataGenerator:
    @staticmethod
    def generate_feature_by_sum(dataset: pd.DataFrame, features: list, new_feature_name: str) -> pd.DataFrame:
        dataset=dataset.copy()
        dataset.loc[:, new_feature_name] = dataset.loc[:, features].sum(axis=1)
        return dataset 
        
    @staticmethod
    def generate_feature_by_mean(dataset: pd.DataFrame, features: list, new_feature_name: str) -> pd.DataFrame:
        dataset=dataset.copy()
        dataset.loc[:, new_feature_name] = dataset.loc[:, features].mean(axis=1).astype(int)
        return dataset 