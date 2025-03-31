import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sklearn.model_selection
from sklearn.preprocessing import LabelEncoder, StandardScaler
import urllib
import PIL
import requests
from datetime import date
import holidays
import os
from tqdm.auto import tqdm
import gc
from catboost import Pool, CatBoostRegressor, cv

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
random_state = np.random.RandomState(714)
rng = np.random.default_rng(714)

def select_within_boundary(df, boundary) -> bool:
    return (
        (df["pickup_longitude"] >= boundary["longitude_min"])
        & (df["pickup_longitude"] <= boundary["longitude_max"])
        & (df["pickup_latitude"] >= boundary["latitude_min"])
        & (df["pickup_latitude"] <= boundary["latitude_max"])
        & (df["dropoff_longitude"] >= boundary["longitude_min"])
        & (df["dropoff_longitude"] <= boundary["longitude_max"])
        & (df["dropoff_latitude"] >= boundary["latitude_min"])
        & (df["dropoff_latitude"] <= boundary["latitude_max"])
    )


def select_in_boundary(df: pd.DataFrame) -> pd.DataFrame:
    boundary = {
        "longitude_min": -74.5,
        "longitude_max": -72.8,
        "latitude_min": 40.5,
        "latitude_max": 41.8,
    }

    return df[select_within_boundary(df, boundary)]


def drop_on_water(df: pd.DataFrame) -> pd.DataFrame:
    def lonlat_to_xy(longitude, latitude, x_range, y_range, boundary):
        longitude_range = boundary["longitude_max"] - boundary["longitude_min"]
        latitude_range = boundary["latitude_max"] - boundary["latitude_min"]

        x = x_range * (longitude - boundary["longitude_min"]) / longitude_range
        y = (
            y_range
            - y_range * (latitude - boundary["latitude_min"]) / latitude_range
        )

        return (x.astype(int), y.astype(int))

    mask_url = urllib.request.urlopen("https://imgur.com/XGHkdoK.png")
    mask = np.array(PIL.Image.open(mask_url))[:, :, 0] > 0.92

    mask = np.c_[mask, np.full([mask.shape[0], 1], False)]
    mask = np.r_[mask, np.full([1, mask.shape[1]], False)]

    boundary = {
        "longitude_min": -74.5,
        "longitude_max": -72.8,
        "latitude_min": 40.5,
        "latitude_max": 41.8,
    }

    pickup_x, pickup_y = lonlat_to_xy(
        df.loc[:, "pickup_longitude"],
        df.loc[:, "pickup_latitude"],
        mask.shape[1] - 1,
        mask.shape[0] - 1,
        boundary,
    )

    dropoff_x, dropoff_y = lonlat_to_xy(
        df.loc[:, "dropoff_longitude"],
        df.loc[:, "dropoff_latitude"],
        mask.shape[1] - 1,
        mask.shape[0] - 1,
        boundary,
    )

    on_land = mask[pickup_y, pickup_x] & mask[dropoff_y, dropoff_x]
    return df[on_land]


def drop_same_pick_drop(df: pd.DataFrame):
    filter = (df["pickup_longitude"] == df["dropoff_longitude"]) & (
        df["pickup_latitude"] == df["dropoff_latitude"]
    )

    return df[~filter]


def drop_nonsense_fareamount(df: pd.DataFrame):
    return df[(df.fare_amount > 0) & (df.fare_amount <= 500)]


def get_lat_lon(df: pd.DataFrame, unit="rad"):
    # Return lat, lon in radian
    lat1 = df["pickup_latitude"].copy().to_numpy()
    lon1 = df["pickup_longitude"].copy().to_numpy()
    lat2 = df["dropoff_latitude"].copy().to_numpy()
    lon2 = df["dropoff_longitude"].copy().to_numpy()

    return lat1, lon1, lat2, lon2


def cal_rotated_coordinate(lat1, lon1, lat2, lon2) -> np.ndarray:
    p1 = np.column_stack([lat1, lon1])
    p2 = np.column_stack([lat2, lon2])

    theta = -np.radians(29).astype("float32")

    rot = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

    # Perform rotate row by row and split
    lat1, lon1 = np.hsplit(np.einsum("ij, mj -> mi", rot, p1), 2)
    lat2, lon2 = np.hsplit(np.einsum("ij, mj -> mi", rot, p2), 2)
    lat1, lon1, lat2, lon2 = map(lambda x: x.ravel(), [lat1, lon1, lat2, lon2])
    return lat1, lon1, lat2, lon2


def get_rotated_coordinate(df: pd.DataFrame):
    lat1, lon1, lat2, lon2 = get_lat_lon(df)
    header = [
        "rotated_pickup_latitude",
        "rotated_pickup_longitude",
        "rotated_dropoff_latitude",
        "rotated_dropoff_longitude",
    ]

    mtx = cal_rotated_coordinate(lat1, lon1, lat2, lon2)

    df_coordinate = pd.DataFrame(mtx, index=header).transpose()

    return pd.concat([df, df_coordinate], axis=1)


def get_euclidean(df: pd.DataFrame):
    lat1, lon1, lat2, lon2 = get_lat_lon(df)
    return (
        np.linalg.norm(
            np.column_stack([lat1, lon1]) - np.column_stack([lat2, lon2]),
            axis=1,
        )
        * 3959
    )


def cal_haversine_distance(lat1, lon1, lat2, lon2):
    dlat = lat1 - lat2
    dlon = lon1 - lon2

    tmp = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )

    return 2 * np.arcsin(np.sqrt(tmp)) * 3959


def get_haversine_distance(df: pd.DataFrame):
    # Return haversine distance in miles
    lat1, lon1, lat2, lon2 = get_lat_lon(df)
    return cal_haversine_distance(lat1, lon1, lat2, lon2)


def get_correct_manhattan(df: pd.DataFrame):
    lat1, lon1, lat2, lon2 = get_lat_lon(df, "mile")
    lat1, lon1, lat2, lon2 = cal_rotated_coordinate(lat1, lon1, lat2, lon2)

    dlat = abs(lat1 - lat2)
    dlon = abs(lon1 - lon2)

    return (dlat + dlon).ravel()


def get_haversine_bearing(df: pd.DataFrame):
    lat1, lon1, lat2, lon2 = get_lat_lon(df)

    dlat = lat1 - lat2
    dlon = lon1 - lon2

    return np.arctan2(
        np.sin(dlon * np.cos(lat2)),
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1) * np.cos(lat2) * np.cos(dlon),
    )


def get_historical_temp_precipitation():
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude=40.71&longitude=-74.01&start_date=2009-01-01&end_date=2015-12-31&hourly=apparent_temperature,precipitation"
    response = requests.get(url)
    data = response.json()

    df_tmp = pd.DataFrame(data["hourly"])
    df_tmp["time"] = pd.to_datetime(df_tmp["time"])

    return df_tmp.set_index("time").to_dict()


def convert_time(x: pd.Series) -> pd.Series:
    date = pd.to_datetime(x.dt.date)
    hour = x.dt.hour

    return pd.Series(date + hour.astype("timedelta64[h]"))


def add_temp_precipitation(df: pd.DataFrame):
    temp_dict = get_historical_temp_precipitation()
    date_time = convert_time(df["pickup_datetime"])
    df["apparent_temperature"] = date_time.map(
        temp_dict["apparent_temperature"]
    ).astype("float32")
    df["precipitation"] = date_time.map(temp_dict["precipitation"]).astype(
        "float32"
    )

    return df


# add time information
def add_time_and_holiday_info(df: pd.DataFrame) -> pd.DataFrame:
    # Add time information
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
    
    df["year"] = df.pickup_datetime.dt.year
    df["month"] = df.pickup_datetime.dt.month.astype("uint8")
    df["day"] = df.pickup_datetime.dt.day.astype("uint8")
    df["weekday"] = df.pickup_datetime.dt.weekday.astype("uint8")
    df["hour"] = df.pickup_datetime.dt.hour.astype("uint8")

    # Add holiday information
#     us_holidays = holidays.US()
#     df["is_holiday"] = df.pickup_datetime.dt.date.isin(us_holidays).astype(
#         "uint8"
#     )

    return df


def distance_to_airport(df):
    """
    JFK: John F. Kennedy International Airport
    EWR: Newark Liberty International Airport
    LGA: LaGuardia Airport
    SOL: Statue of Liberty
    NYC: Newyork Central
    """
    jfk_coord = np.radians((40.639722, -73.778889))
    ewr_coord = np.radians((40.6925, -74.168611))
    lga_coord = np.radians((40.77725, -73.872611))
    sol_coord = np.radians((40.6892, -74.0445))  # Statue of Liberty
    nyc_coord = np.radians((40.7141667, -74.0063889))

    pickup_lat = df["pickup_latitude"]
    dropoff_lat = df["dropoff_latitude"]
    pickup_lon = df["pickup_longitude"]
    dropoff_lon = df["dropoff_longitude"]

    pickup_jfk = cal_haversine_distance(
        pickup_lat, pickup_lon, jfk_coord[0], jfk_coord[1]
    )
    dropoff_jfk = cal_haversine_distance(
        jfk_coord[0], jfk_coord[1], dropoff_lat, dropoff_lon
    )
    pickup_ewr = cal_haversine_distance(
        pickup_lat, pickup_lon, ewr_coord[0], ewr_coord[1]
    )
    dropoff_ewr = cal_haversine_distance(
        ewr_coord[0], ewr_coord[1], dropoff_lat, dropoff_lon
    )
    pickup_lga = cal_haversine_distance(
        pickup_lat, pickup_lon, lga_coord[0], lga_coord[1]
    )
    dropoff_lga = cal_haversine_distance(
        lga_coord[0], lga_coord[1], dropoff_lat, dropoff_lon
    )
    pickup_sol = cal_haversine_distance(
        pickup_lat, pickup_lon, sol_coord[0], sol_coord[1]
    )
    dropoff_sol = cal_haversine_distance(
        sol_coord[0], sol_coord[1], dropoff_lat, dropoff_lon
    )
    pickup_nyc = cal_haversine_distance(
        pickup_lat, pickup_lon, nyc_coord[0], nyc_coord[1]
    )
    dropoff_nyc = cal_haversine_distance(
        nyc_coord[0], nyc_coord[1], dropoff_lat, dropoff_lon
    )

    df["jfk_dist"] = (pickup_jfk + dropoff_jfk).astype("float32")
    df["ewr_dist"] = (pickup_ewr + dropoff_ewr).astype("float32")
    df["lga_dist"] = (pickup_lga + dropoff_lga).astype("float32")
    df["sol_dist"] = (pickup_sol + dropoff_sol).astype("float32")
    df["nyc_dist"] = (pickup_nyc + dropoff_nyc).astype("float32")

    return df


def date_format(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_time = df["pickup_datetime"].copy()

    date_time = date_time.str.slice(0, 16)
    date_time = pd.to_datetime(date_time, utc=True, format="%Y-%m-%d %H:%M")

    df["pickup_datetime"] = date_time
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop negative fare amount
    if 'fare_amount' in df.columns:
        df = drop_nonsense_fareamount(df)
    # Drop nan value
    df = df.dropna()
    # Drop data out of boundary
    df = select_in_boundary(df)
    # Drop data on water
    # df = drop_on_water(df)
    # Drop same pickup and dropoff data
    # df = drop_same_pick_drop(df)
    df = df.reset_index(drop=True)
    return df


def engineering(df: pd.DataFrame):
    df = df.copy()
    df.loc[:, "pickup_longitude"] = np.radians(df.loc[:, "pickup_longitude"])
    df.loc[:, "pickup_latitude"] = np.radians(df.loc[:, "pickup_latitude"])
    df.loc[:, "dropoff_longitude"] = np.radians(df.loc[:, "dropoff_longitude"])
    df.loc[:, "dropoff_latitude"] = np.radians(df.loc[:, "dropoff_latitude"])
    df.loc[:, "haversine"] = get_haversine_distance(df)
    df.loc[:, "haversine_bearing"] = get_haversine_bearing(df)
    df = add_time_and_holiday_info(df)
    df = distance_to_airport(df)
    df = df.drop(columns=["pickup_datetime"])

    return df





train_path = "data/new-york-city-taxi-fare-prediction/train_20000000.feather"
test_path = "data/new-york-city-taxi-fare-prediction/test.csv"
key_path = "data/new-york-city-taxi-fare-prediction/test.csv"

df_key = pd.read_csv(key_path, usecols=["key"])
data_train = pd.read_feather(train_path)
data_test = pd.read_csv(test_path)

print("Engineering...")
df_train = clean_data(data_train)
df_train = engineering(df_train)
# df_test = clean_data(data_test)
df_test = engineering(data_test)
df_train = df_train.drop(columns=['key'])
df_test = df_test.drop(columns=['key'])

del train_path, test_path, key_path, data_train
gc.collect()

cat_feature = ["year", "month", "day", "weekday", "hour"]

def random_split(df: pd.DataFrame):
    x = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(
        x, y, train_size=0.7, random_state=random_state
    )

    return x_train, x_val, y_train, y_val

use_cols = [
    'pickup_longitude', 
    'pickup_latitude',
    'dropoff_longitude', 
    'dropoff_latitude', 
#     'passenger_count', 
#     'euclidean',
    'haversine', 
    'haversine_bearing', 
#     'correct_manhattan', 
    'year', 
    'month',
    'day',
    'weekday', 
    'hour', 
    'jfk_dist', 
    'ewr_dist', 
    'lga_dist',
    'sol_dist', 
    'nyc_dist'
]



x_train, x_val, y_train, y_val = random_split(df_train)
x_train = x_train[use_cols]
x_val = x_val[use_cols]
del df_train
gc.collect()

pool_train = Pool(x_train, y_train, cat_features=cat_feature)
pool_val = Pool(x_val, y_val, cat_features=cat_feature)
pool_test = Pool(df_test, cat_features=cat_feature)

del x_train, y_train, x_val, y_val
gc.collect()



pool_train.get_feature_names()

print("Modeling...")
model = CatBoostRegressor(
    iterations=1000,
    depth=6,
    learning_rate=0.1,
    loss_function="RMSEWithUncertainty",
    use_best_model=True,
    early_stopping_rounds=500,
    eval_metric="RMSE",
    random_seed=714,
    verbose=1000,
    task_type="GPU",
    devices="0:1",
    od_type="Iter",
    l2_leaf_reg=3.4,
    per_float_feature_quantization=['4:border_count=1024'],
    random_strength=0.8,
    border_count=128,
    leaf_estimation_iterations=5
)



model.fit(pool_train, eval_set=pool_val)
gc.collect()



y_pred = model.predict(pool_test)
df_out = pd.concat([df_key, pd.Series(y_pred[:,0], name="fare_amount")], axis=1)
df_out.to_csv("prediction_20_000_000.csv", index=False)