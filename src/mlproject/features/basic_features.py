import pandas as pd
import numpy as np

def convert_datetime(data, cols):
    for c in cols:
        data[c] = pd.to_datetime(data[c], errors="coerce")
    return data

def add_distance_km(data):
    R = 6371 

    lat1 = np.radians(data["pickup_latitude"])
    lon1 = np.radians(data["pickup_longitude"])
    lat2 = np.radians(data["dropoff_latitude"])
    lon2 = np.radians(data["dropoff_longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    data["distance_km"] = R * c
    return data

def add_pickup_month(data):
    data["pickup_month"] = data["pickup_datetime"].dt.month
    return data

def add_pickup_hour(data):
    data["pickup_hour"] = data["pickup_datetime"].dt.hour
    return data

def add_pickup_weekday(data):
    data["pickup_weekday"] = data["pickup_datetime"].dt.weekday
    return data

def drop_large_distance(data, max_km=200):
    return data[data["distance_km"] <= max_km].reset_index(drop=True)
