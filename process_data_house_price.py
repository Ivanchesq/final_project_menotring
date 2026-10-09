import numpy as np
import pandas as pd


def _mode(series, fallback=None):
    modes = series.mode(dropna=True)
    if modes.empty:
        return fallback
    return modes.iloc[0]


def _process_garage(df, stats=None):
    data = df.copy()
    cat_cols = ["GarageFinish", "GarageQual", "GarageCond"]

    no_garage = data["GarageType"].isna()
    data.loc[no_garage, ["GarageCars", "GarageArea"]] = 0
    data.loc[no_garage, "GarageYrBlt"] = 0
    data.loc[no_garage, "GarageType"] = "None"
    for col in cat_cols:
        data.loc[no_garage, col] = "None"

    has_garage = data["GarageType"] != "None"
    if stats is None:
        present = data.loc[has_garage]
        stats = {
            "cars_by_type": present.groupby("GarageType")["GarageCars"].median(),
            "area_by_type": present.groupby("GarageType")["GarageArea"].median(),
            "cars_global": present["GarageCars"].median(),
            "area_global": present["GarageArea"].median(),
        }
        for col in cat_cols:
            stats[col] = _mode(present[col], "None")

    data.loc[has_garage, "GarageCars"] = (
        data.loc[has_garage, "GarageCars"]
        .fillna(data.loc[has_garage, "GarageType"].map(stats["cars_by_type"]))
        .fillna(stats["cars_global"])
    )
    data.loc[has_garage, "GarageArea"] = (
        data.loc[has_garage, "GarageArea"]
        .fillna(data.loc[has_garage, "GarageType"].map(stats["area_by_type"]))
        .fillna(stats["area_global"])
    )

    missing_year = has_garage & data["GarageYrBlt"].isna()
    data.loc[missing_year, "GarageYrBlt"] = data.loc[missing_year, "YearBuilt"]
    bad_year = has_garage & (data["GarageYrBlt"] > data["YrSold"])
    data.loc[bad_year, "GarageYrBlt"] = data.loc[bad_year, "YearBuilt"]

    for col in cat_cols:
        data.loc[has_garage, col] = data.loc[has_garage, col].fillna(stats[col])

    data["Area_Per_Car"] = np.where(
        data["GarageCars"] > 0,
        data["GarageArea"] / data["GarageCars"],
        0,
    )
    return data, stats


def _process_basement(df, stats=None):
    data = df.copy()
    num_cols = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"]
    cat_cols = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]

    for col in num_cols:
        data[col] = data[col].fillna(0)

    no_bsmt = data["TotalBsmtSF"] == 0
    for col in cat_cols:
        data.loc[no_bsmt, col] = "None"

    has_bsmt = ~no_bsmt
    if stats is None:
        present = data.loc[has_bsmt]
        stats = {col: _mode(present[col], "None") for col in cat_cols}
        second = present.loc[present["BsmtFinSF2"] > 0, "BsmtFinType2"].dropna()
        stats["BsmtFinType2_if_sf2"] = _mode(second, stats["BsmtFinType2"])

    no_second = has_bsmt & data["BsmtFinType2"].isna() & (data["BsmtFinSF2"] == 0)
    data.loc[no_second, "BsmtFinType2"] = "Unf"
    missing_second = has_bsmt & data["BsmtFinType2"].isna() & (data["BsmtFinSF2"] > 0)
    data.loc[missing_second, "BsmtFinType2"] = stats["BsmtFinType2_if_sf2"]

    for col in ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1"]:
        data.loc[has_bsmt, col] = data.loc[has_bsmt, col].fillna(stats[col])

    data["Has_Bsmt"] = has_bsmt.astype(int)
    data["Bsmt_Fin_SF"] = data["BsmtFinSF1"] + data["BsmtFinSF2"]
    data["Bsmt_Baths"] = data["BsmtFullBath"] + 0.5 * data["BsmtHalfBath"]
    return data.drop(columns=["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "BsmtFullBath", "BsmtHalfBath"]), stats


def _process_masvnr(df, vnr_mode):
    data = df.copy()
    data["MasVnrArea"] = data["MasVnrArea"].fillna(0)
    data["MasVnrType"] = data["MasVnrType"].fillna("None")
    data.loc[data["MasVnrArea"] == 0, "MasVnrType"] = "None"
    data.loc[(data["MasVnrArea"] > 0) & (data["MasVnrType"] == "None"), "MasVnrType"] = vnr_mode
    data["Has_MasVnr"] = (data["MasVnrArea"] > 0).astype(int)
    return data


def _drop_living_area_outliers(df):
    price = df["SalePrice"].to_numpy(float)
    area = df["GrLivArea"].to_numpy(float)
    slope, intercept = np.polyfit(area, price, 1)
    resid = price - (slope * area + intercept)
    sigma = resid.std(ddof=2)
    leverage = 1 / len(area) + (area - area.mean()) ** 2 / np.sum((area - area.mean()) ** 2)
    studentized = resid / (sigma * np.sqrt(1 - leverage))
    return df.loc[studentized >= -4].copy()


def process_data(file_path, is_train=True, stats=None):
    if not is_train and stats is None:
        raise ValueError("Для test нужны статистики, посчитанные на train")

    df = pd.read_csv(file_path)
    if is_train:
        stats = {}

    df["MSSubClass"] = df["MSSubClass"].astype(str)

    if is_train:
        stats["mszoning_by_pair"] = df.groupby(["Neighborhood", "MSSubClass"])["MSZoning"].agg(_mode)
        stats["mszoning_by_nb"] = df.groupby("Neighborhood")["MSZoning"].agg(_mode)
        stats["mszoning_global"] = _mode(df["MSZoning"])
    pair_key = pd.MultiIndex.from_arrays([df["Neighborhood"], df["MSSubClass"]])
    pair_fill = pd.Series(pair_key.map(stats["mszoning_by_pair"]), index=df.index)
    df["MSZoning"] = (
        df["MSZoning"]
        .fillna(pair_fill)
        .fillna(df["Neighborhood"].map(stats["mszoning_by_nb"]))
        .fillna(stats["mszoning_global"])
    )

    df["LotFrontage"] = df["LotFrontage"].clip(upper=200)
    if is_train:
        stats["lot_l1"] = df.groupby(["Neighborhood", "LotConfig"])["LotFrontage"].median()
        stats["lot_l2"] = df.groupby("Neighborhood")["LotFrontage"].median()
        stats["lot_global"] = df["LotFrontage"].median()
    lot_l1 = pd.Series(df.set_index(["Neighborhood", "LotConfig"]).index.map(stats["lot_l1"]), index=df.index)
    df["LotFrontage"] = df["LotFrontage"].fillna(lot_l1).fillna(df["Neighborhood"].map(stats["lot_l2"])).fillna(stats["lot_global"])
    df["LotArea"] = np.log1p(df["LotArea"])

    df = df.drop(columns=["Street", "Utilities"])
    df["Alley"] = df["Alley"].fillna("None")
    df["LotConfig"] = df["LotConfig"].replace({"FR2": "FR", "FR3": "FR"})

    railroad = ["RRNn", "RRAn", "RRNe", "RRAe"]
    arterial = ["Artery", "Feedr"]
    park = ["PosN", "PosA"]
    df["RailRoad"] = (df["Condition1"].isin(railroad) | df["Condition2"].isin(railroad)).astype(int)
    df["NearStreet"] = (df["Condition1"].isin(arterial) | df["Condition2"].isin(arterial)).astype(int)
    df["NearPark"] = (df["Condition1"].isin(park) | df["Condition2"].isin(park)).astype(int)
    df = df.drop(columns=["Condition1", "Condition2"])

    style_map = {
        "1Story": "1Story",
        "1.5Fin": "1.5Story",
        "1.5Unf": "1.5Story",
        "2Story": "2Story",
        "2.5Fin": "2.5Story",
        "2.5Unf": "2.5Story",
        "SFoyer": "Split",
        "SLvl": "Split",
    }
    df["HouseStyle_Group"] = df["HouseStyle"].map(style_map)
    df = df.drop(columns=["HouseStyle"])

    df, garage_stats = _process_garage(df, None if is_train else stats["garage"])
    if is_train:
        stats["garage"] = garage_stats

    df["House_Age"] = np.maximum(0, df["YrSold"] - df["YearBuilt"])
    df["Remod_Age"] = np.maximum(0, df["YrSold"] - df["YearRemodAdd"])
    df["Is_Remodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
    df["Age_At_Remod"] = df["YearRemodAdd"] - df["YearBuilt"]
    df = df.drop(columns=["YearBuilt", "YearRemodAdd"])

    df["RoofStyle"] = df["RoofStyle"].map({"Gable": "Gable", "Hip": "Hip"}).fillna("Other")
    df = df.drop(columns=["RoofMatl"])

    if is_train:
        stats["ext1_global"] = _mode(df["Exterior1st"])
        stats["ext2_global"] = _mode(df["Exterior2nd"])
        stats["ext1_by_nb"] = df.groupby("Neighborhood")["Exterior1st"].agg(lambda s: _mode(s, stats["ext1_global"]))
        stats["ext2_by_nb"] = df.groupby("Neighborhood")["Exterior2nd"].agg(lambda s: _mode(s, stats["ext2_global"]))
    df["Exterior1st"] = df["Exterior1st"].fillna(df["Neighborhood"].map(stats["ext1_by_nb"])).fillna(stats["ext1_global"])
    df["Exterior2nd"] = df["Exterior2nd"].fillna(df["Neighborhood"].map(stats["ext2_by_nb"])).fillna(stats["ext2_global"])
    df["Exterior2nd"] = df["Exterior2nd"].replace({"CmentBd": "CemntBd", "Wd Shng": "WdShing", "Brk Cmn": "BrkComm"})
    df["Is_Multiple_Exterior"] = (df["Exterior1st"] != df["Exterior2nd"]).astype(int)
    top_materials = ["VinylSd", "MetalSd", "HdBoard", "Wd Sdng", "Plywood", "CemntBd", "WdShing"]
    df["Exterior1st"] = df["Exterior1st"].where(df["Exterior1st"].isin(top_materials), "Other")
    df["Exterior2nd"] = df["Exterior2nd"].where(df["Exterior2nd"].isin(top_materials), "Other")

    if is_train:
        stats["vnr_mode"] = _mode(df.loc[df["MasVnrArea"] > 0, "MasVnrType"])
    df = _process_masvnr(df, stats["vnr_mode"])

    df, bsmt_stats = _process_basement(df, None if is_train else stats["bsmt"])
    if is_train:
        stats["bsmt"] = bsmt_stats

    df["Is_GasA"] = (df["Heating"] == "GasA").astype(int)
    df = df.drop(columns=["Heating"])
    df["CentralAir"] = (df["CentralAir"] == "Y").astype(int)

    if is_train:
        stats["elec_mode"] = _mode(df["Electrical"])
    df["Has_SBrkr"] = (df["Electrical"].fillna(stats["elec_mode"]) == "SBrkr").astype(int)
    df = df.drop(columns=["Electrical"])

    df["Total_SF"] = df["GrLivArea"] + df["TotalBsmtSF"]
    df["Has_2ndFlr"] = (df["2ndFlrSF"] > 0).astype(int)
    df["Has_LowQualFin"] = (df["LowQualFinSF"] > 0).astype(int)
    df = df.drop(columns=["LowQualFinSF"])

    if is_train:
        df = _drop_living_area_outliers(df)

    df["Avg_Room_SF"] = df["GrLivArea"] / df["TotRmsAbvGrd"]
    df["AbvGrd_Baths"] = df["FullBath"] + 0.5 * df["HalfBath"]
    df["Total_Baths"] = df["AbvGrd_Baths"] + df["Bsmt_Baths"]
    df = df.drop(columns=["FullBath", "HalfBath"])
    df["Other_Rms_AbvGrd"] = df["TotRmsAbvGrd"] - df["BedroomAbvGr"] - df["KitchenAbvGr"]

    if is_train:
        stats["kitchen_global"] = _mode(df["KitchenQual"])
        stats["kitchen_by_nb"] = df.groupby("Neighborhood")["KitchenQual"].agg(
            lambda s: _mode(s, stats["kitchen_global"])
        )
    df["KitchenQual"] = df["KitchenQual"].fillna(df["Neighborhood"].map(stats["kitchen_by_nb"])).fillna(stats["kitchen_global"])

    df["Fireplaces"] = df["Fireplaces"].clip(upper=2)
    df["FireplaceQu"] = df["FireplaceQu"].fillna("None")
    df.loc[df["Fireplaces"] == 0, "FireplaceQu"] = "None"

    df["Garage_Age"] = np.where(
        df["GarageYrBlt"] > 0,
        np.maximum(0, df["YrSold"] - df["GarageYrBlt"]),
        0,
    )
    df = df.drop(columns=["GarageYrBlt"])

    porch_cols = ["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
    df["Total_Porch_SF"] = df[porch_cols].sum(axis=1)
    df = df.drop(columns=["EnclosedPorch", "3SsnPorch", "ScreenPorch"])

    df["Has_Pool"] = (df["PoolArea"] > 0).astype(int)
    df = df.drop(columns=["PoolArea", "PoolQC"])
    df["Fence"] = df["Fence"].fillna("None")
    df["Has_Shed"] = (df["MiscFeature"] == "Shed").astype(int)
    df = df.drop(columns=["MiscFeature", "MiscVal"])

    if is_train:
        stats["func_global"] = _mode(df["Functional"])
        stats["func_by_nb"] = df.groupby("Neighborhood")["Functional"].agg(lambda s: _mode(s, stats["func_global"]))
    df["Functional"] = df["Functional"].fillna(df["Neighborhood"].map(stats["func_by_nb"])).fillna(stats["func_global"])

    df["MoSold_Sin"] = np.sin(2 * np.pi * df["MoSold"] / 12)
    df["MoSold_Cos"] = np.cos(2 * np.pi * df["MoSold"] / 12)
    df["Is_Peak_Season"] = df["MoSold"].isin([5, 6, 7]).astype(int)
    df = df.drop(columns=["MoSold"])

    if is_train:
        stats["sale_type_by_year"] = df.groupby("YrSold")["SaleType"].agg(_mode)
        stats["sale_type_global"] = _mode(df["SaleType"])
    df["SaleType"] = df["SaleType"].fillna(df["YrSold"].map(stats["sale_type_by_year"])).fillna(stats["sale_type_global"])

    if "SalePrice" in df.columns:
        df["SalePrice"] = np.log1p(df["SalePrice"])

    df = df.drop(columns=["Id"])

    if is_train:
        return df, stats
    return df
