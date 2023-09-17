import pandas sd pd

class DataCleaner:
    @staticmethod
    def drop_feature(dataset: pd.DataFrame, columns: list) -> pd.DataFrame:
        dataset = dataset.copy()
        dataset = dataset.drop(columns=columns)
        return dataset

    @staticmethod
    def change_feature_type(dataset: pd.DataFrame, mapper: dict) -> pd.DataFrame:
        dataset = dataset.copy()
        dataset = dataset.astype(mapper)
        return dataset
    
    @staticmethod
    def impute_missing_numerical_feature(dataset: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        numerical_features = dataset.select_dtypes(include=['int', 'float']).columns.tolist()        
        if dataset_type == 'train':
            # fill missing numerical features with mean
            imputer = SimpleImputer(strategy='mean')
            dataset.loc[:, numerical_features] = imputer.fit_transform(dataset.loc[:, numerical_features])
            pickle.dump(imputer, open(os.getenv('NUMERICAL_IMPUTER_PATH'), "wb"))
            
        elif dataset_type == 'test':
            # fill missing numerical features with saved mean imputer
            imputer = pickle.load(open(os.getenv('NUMERICAL_IMPUTER_PATH'), "rb"))
            dataset.loc[:, numerical_features] = imputer.fit_transform(dataset.loc[:, numerical_features])
        else:
            print("dataset_type is neither train or test")
        return dataset

    @staticmethod
    def impute_missing_categorical_feature(
        dataset: pd.DataFrame, dataset_type: str, 
        categorical_features_missing_on_purpose: list = [
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
            'BsmtFinType2','GarageType', 'GarageFinish', 'GarageQual',
            'GarageCond','Alley', 'MasVnrType', 'Fence', 'FireplaceQu', 
            'MiscFeature', 'PoolQC'
        ]) -> pd.DataFrame:

        categorical_features = dataset.select_dtypes(include=['object']).columns.tolist()

        # fill expected missing categorical values
        imputer = SimpleImputer(strategy='constant', fill_value='Missing')
        dataset.loc[:, categorical_features_missing_on_purpose] = imputer.fit_transform(dataset.loc[:, categorical_features_missing_on_purpose])

        if dataset_type == 'train':
            # fill missing categorical features with mode
            imputer = SimpleImputer(strategy='most_frequent')
            dataset.loc[:, categorical_features] = imputer.fit_transform(dataset.loc[:, categorical_features])
            pickle.dump(imputer, open(os.getenv('CATEGORICAL_IMPUTER_PATH'), 'wb'))
            
        elif dataset_type == 'test':
            # fill missing categorical features with saved mode imputer
            imputer = pickle.load(open(os.getenv('CATEGORICAL_IMPUTER_PATH'), "rb"))
            dataset.loc[:, categorical_features] = imputer.fit_transform(dataset.loc[:, categorical_features])
        else:
            print("dataset_type is neither train or test")
            
        return dataset