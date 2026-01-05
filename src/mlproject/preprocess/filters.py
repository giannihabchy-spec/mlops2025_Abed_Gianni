def drop_missing_rows(data):
    return data.dropna().reset_index(drop=True)

def filter_trip_duration(data, min_sec=60, max_sec=10800):
    return data[
        (data["trip_duration"] >= min_sec) &
        (data["trip_duration"] <= max_sec)
    ].reset_index(drop=True)

def drop_duplicate_rows(data):
    return data.drop_duplicates(keep="first").reset_index(drop=True)

def drop_same_location(data, tol=1e-6):
    return data[
        (abs(data["pickup_latitude"] - data["dropoff_latitude"]) > tol) |
        (abs(data["pickup_longitude"] - data["dropoff_longitude"]) > tol)
    ].reset_index(drop=True)
