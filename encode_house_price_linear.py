import pandas as pd
from sklearn.preprocessing import OneHotEncoder


ORDINAL_MAPS = {
    "LotShape": {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
    "LandSlope": {"Sev": 0, "Mod": 1, "Gtl": 2},
    "ExterQual": {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "ExterCond": {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "HeatingQC": {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "KitchenQual": {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "BsmtQual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtCond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "FireplaceQu": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageQual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageCond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "Functional": {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "Fence": {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
}

OHE_COLS = [
    "MSSubClass",
    "MSZoning",
    "Alley",
    "LandContour",
    "LotConfig",
    "BldgType",
    "RoofStyle",
    "HouseStyle_Group",
    "MasVnrType",
    "Foundation",
    "GarageType",
    "SaleType",
    "SaleCondition",
    "Exterior1st",
    "Exterior2nd"

]
NUM_COLS = [
    "LotFrontage",
    "LotArea",
    "MasVnrArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "GrLivArea",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "Garage_Age",
    "GarageCars",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
    "YrSold",
    "Area_Per_Car",
    "House_Age",
    "Remod_Age",
    "Age_At_Remod",
    "Bsmt_Fin_SF",
    "Bsmt_Baths",
    "Total_SF",
    "Avg_Room_SF",
    "AbvGrd_Baths",
    "Total_Baths",
    "Other_Rms_AbvGrd",
    "Total_Porch_SF",
    "MoSold_Sin",
    "MoSold_Cos",
    "OverallQual",
    "OverallCond",
]
BINARY_COLS = [
    "RailRoad",
    "NearStreet",
    "NearPark",
    "Is_Remodeled",
    "Is_Multiple_Exterior",
    "Has_MasVnr",
    "Has_Bsmt",
    "Is_GasA",
    "Has_SBrkr",
    "Has_2ndFlr",
    "Has_LowQualFin",
    "Has_Pool",
    "Has_Shed",
    "Is_Peak_Season",
    "CentralAir",
]
FREQ_COLS = ["Neighborhood"]
ORDINAL_COLS = list(ORDINAL_MAPS)


def _map_ordinal(df):
    data = df.copy()
    for col, mapping in ORDINAL_MAPS.items():
        missed = sorted(data.loc[data[col].map(mapping).isna() & data[col].notna(), col].astype(str).unique())
        if missed:
            raise ValueError(f"{col}: нет уровня в маппинге: {missed}")
        data[col] = data[col].map(mapping)
    return data


def encode_linear(df, is_train=True, stats=None):
    if not is_train and stats is None:
        raise ValueError("Для test нужны статистики, посчитанные на train")

    data = _map_ordinal(df)
    if is_train:
        stats = {} if stats is None else stats
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded = encoder.fit_transform(data[OHE_COLS])
        stats["ohe"] = encoder
        stats["freq"] = {col: data[col].value_counts(normalize=True) for col in FREQ_COLS}
    else:
        encoder = stats["ohe"]
        encoded = encoder.transform(data[OHE_COLS])

    ohe_frame = pd.DataFrame(
        encoded,
        index=data.index,
        columns=encoder.get_feature_names_out(OHE_COLS),
    )
    for col in FREQ_COLS:
        data[col] = data[col].map(stats["freq"][col]).fillna(0)

    sale_price = data.pop("SalePrice") if "SalePrice" in data.columns else None
    keep = ORDINAL_COLS + NUM_COLS + BINARY_COLS + FREQ_COLS
    frame = pd.concat([data[keep], ohe_frame], axis=1)
    if sale_price is not None:
        frame["SalePrice"] = sale_price.to_numpy()

    if is_train:
        return frame, stats
    return frame
