from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler


class DataFeatureTransformer:

    def log_transform_feature(dataset: pd.DataFrame) -> pd.DataFrame:
        dataset = dataset.copy()
        try:
            numerical_columns = dataset.select_dtypes(include=['int', 'float']).columns
            dataset.loc[:, numerical_columns] = np.log1p(dataset.loc[:, numerical_columns])
        except Exception as e:
            print(e)
        return dataset

    def ordinal_encoding(dataset: pd.DataFrame) -> pd.DataFrame:
        dataset = dataset.copy()
        ordinal_categorical_columns ={
            "ExterQual": ["Ex", "Gd", "TA", "Fa", "Po"], 
            "ExterCond": ["Ex", "Gd", "TA", "Fa", "Po"], 
            "BsmtQual":  ["Ex", "Gd", "TA", "Fa", "Po", "Missing"],
            "BsmtCond": ["Ex", "Gd", "TA", "Fa", "Po", "Missing"],
            "BsmtExposure": ["Gd", "Av", "Mn", "No", "Missing"],
            "BsmtFinType1": ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "Missing"],
            "BsmtFinType2": ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "Missing"],
            "HeatingQC": ["Ex", "Gd", "TA", "Fa", "Po"], 
            "KitchenQual": ["Ex", "Gd", "TA", "Fa", "Po"], 
            "FireplaceQu": ["Ex", "Gd", "TA", "Fa", "Po", "Missing"],
            "GarageFinish": ["Fin", "RFn", "Unf", "Missing"], 
            "GarageQual": ["Ex", "Gd", "TA", "Fa", "Po", "Missing"], 
            "GarageCond": ["Ex", "Gd", "TA", "Fa", "Po", "Missing"], 
            "PoolQC": ["Ex", "Gd", "TA", "Fa", "Missing"], 
            "Fence": ["GdPrv", "MnPrv", "GdWo", "MnWw", "Missing"]
        } # gather ordinal categorical column
       
        ### Categorical columns transformation
        for f, v in ordinal_categorical_columns.items():
            if f in dataset.columns:
                ordinal_encoder = OrdinalEncoder(categories=[v]) # define ordinal encoder
                dataset[f] = ordinal_encoder.fit_transform(dataset[[f]]).astype(int) # ordinal encoding
        return dataset
    
    def one_hot_encoding(dataset: pd.DataFrame) -> pd.DataFrame:
        one_hot_categorical_columns = {
            "MSSubClass": ["20", "30", "40", "45",	"50", "60", "70", "75", "80", "85", "90", "120", "150", "160", "180", "190"],
            "MSZoning": ["A", "C", "FV", "I", "RH", "RL", "RP", "RM"],
            "Street": ["Pave", "Grvl"],
            "Alley": ["Missing", "Grvl", "Pave"],
            "LotShape": ["Reg", "IR1", "IR2", "IR3"],
            "LandContour": ["Lvl", "Bnk", "Low", "HLS"],
            "Utilities": ["AllPub", "NoSewr", "NoSeWa", "ELO"],
            "LotConfig": ["Inside", "FR2", "Corner", "CulDSac", "FR3"],
            "LandSlope": ["Gtl", "Mod", "Sev"],
            "Neighborhood": ["CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel", "Somerst", "NWAmes", "OldTown", "BrkSide", "Sawyer", "NridgHt", "NAmes", "SawyerW", "IDOTRR", "MeadowV", "Edwards", "Timber", "Gilbert", "StoneBr", "ClearCr", "NPkVill", "Blmngtn", "BrDale", "SWISU", "Blueste"],
            "Condition1": ["Norm", "Feedr", "PosN", "Artery", "RRAe", "RRNn", "RRAn", "PosA", "RRNe"],
            "Condition2": ["Norm", "Artery", "RRNn", "Feedr", "PosN", "PosA", "RRAn", "RRAe", "RRNe"],
            "BldgType": ["1Fam", "2fmCon", "Duplex", "TwnhsE", "Twnhs"],
            "HouseStyle": ["2Story", "1Story", "1.5Fin", "1.5Unf", "SFoyer", "SLvl", "2.5Unf", "2.5Fin"],
            "RoofStyle": ["Gable", "Hip", "Gambrel", "Mansard", "Flat", "Shed"],
            "RoofMatl": ["CompShg", "WdShngl", "Metal", "WdShake", "Membran", "Tar&Grv", "Roll", "ClyTile"],
            "Exterior1st": ["AsbShng", "AsphShn", "BrkComm", "BrkFace", "CBlock", "CemntBd", "HdBoard",	"ImStucc", "MetalSd", "Other", "Plywood", "PreCast", "Stone", "Stucco", "VinylSd", "Wd Sdng", "WdShing"],
            "Exterior2nd": ["AsbShng", "AsphShn", "BrkComm", "BrkFace", "CBlock", "CemntBd", "HdBoard",	"ImStucc", "MetalSd", "Other", "Plywood", "PreCast", "Stone", "Stucco", "VinylSd", "Wd Sdng", "WdShing"],
            "MasVnrType": ["BrkFace", "Missing", "Stone", "BrkCmn", "CBlock"],
            "Foundation": ["PConc", "CBlock", "BrkTil", "Wood", "Slab", "Stone"],
            "Heating": ["GasA", "GasW", "Grav", "Wall", "OthW", "Floor"],
            "CentralAir": ["Y", "N"],
            "Electrical": ["SBrkr", "FuseF", "FuseA", "FuseP", "Mix", "Missing"],
            "Functional": ["Typ", "Min1", "Maj1", "Min2", "Mod", "Maj2", "Sev", "Sal"],
            "GarageType": ["Attchd", "Detchd", "BuiltIn", "CarPort", "Missing", "Basment", "2Types"],
            "PavedDrive": ["Y", "N", "P"],
            "MiscFeature": ["Missing", "Shed", "Gar2", "Othr", "TenC", "Elev"],
            "SaleType": ["WD", "New", "COD", "ConLD", "ConLI", "CWD", "ConLw", "Con", "Oth", "VWD"],
            "SaleCondition": ["Normal", "Abnorml", "Partial", "AdjLand", "Alloca", "Family"]
        } # one hot encoder column

        for f, v in one_hot_categorical_columns.items():
            if f in dataset.columns:
                one_hot_encoder = OneHotEncoder(categories=[v], sparse_output=False, handle_unknown='ignore') # one hot encoder
                dataset_encoded = pd.DataFrame(one_hot_encoder.fit_transform(dataset[[f]])) # one hot encoding
                dataset_encoded.columns = one_hot_encoder.get_feature_names_out([f])
                dataset.drop(columns=[f], inplace=True)
                dataset = pd.concat([dataset, dataset_encoded], axis=1)
        return dataset
    
    def scale_data(dataset: pd.DataFrame, scaler_name: str, dataset_type: str) -> pd.DataFrame:
        dataset = dataset.copy()
        if dataset_type == "train":
            if scaler_name == "standard":
                scaler = StandardScaler()
            elif scaler_name == "minmax":
                scaler = MinMaxScaler()
            elif scaler_name == "robust":
                scaler = RobustScaler()

            dataset_scaled = scaler.fit_transform(dataset) # scale the data    
            dataset = pd.DataFrame(data=dataset_scaled, columns=dataset.columns) # create pandas dataframe
            pickle.dump(scaler, open(os.getenv('SCALER_PATH'), 'wb')) # save the scaler

        elif dataset_type == "test":
            scaler = pickle.load(open(os.getenv('SCALER_PATH'), 'rb'))
            dataset_scaled = scaler.fit_transform(dataset) # scale the data    
            dataset = pd.DataFrame(data=dataset_scaled, columns=dataset.columns) # create pandas dataframe
        return dataset