import sqlite3
from tqdm import tqdm
from config import settings
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from scipy import stats
from sklearn.preprocessing import StandardScaler
from functools import lru_cache
from datetime import datetime, timedelta


@lru_cache(maxsize=1)
def get_connection():
    conn = sqlite3.connect("./data/studentlifethird.sqlite")
    c = conn.cursor()
    conn.execute("PRAGMA journal_mode=OFF;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("PRAGMA cache_size=-2000000;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA mmap_size=30000000000;")
    conn.execute("PRAGMA page_size=4096;")
    return conn, c


def fetch_z_stress(uids: list[str] = None):
    # Base query
    query = "SELECT uid, day, stress, mean, std_dev, z_score FROM stress_z_score"

    # If uids is provided, modify the query to filter by user IDs
    if uids:
        query += " WHERE uid IN ({})".format(
            ", ".join("'" + str(x) + "'" for x in uids)
        )
    # query = query + " LIMIT 10"
    # print(query)  # For debugging purposes

    # Get database connection
    conn, c = get_connection()

    # Execute the query
    c.execute(query)

    # Fetch all results
    results = c.fetchall()

    # Close the connection (optional, but good practice)

    return results


def fetch_mental_pressure(uids: list[str] = None):
    # Base query
    query = """SELECT "uid", "day", "phq4-1", "phq4-2", "phq4-3", "phq4-4", "sse3-1", "sse3-2", "sse3-3", "sse3-4", "stress" FROM stress"""

    # If uids is provided, modify the query to filter by user IDs
    if uids:
        query += " WHERE uid IN ({})".format(
            ", ".join("'" + str(x) + "'" for x in uids)
        )

    # Get database connection
    conn, c = get_connection()

    # Execute the query
    c.execute(query)

    # Fetch all results
    results = c.fetchall()

    return results


def fetch_data(uids: list[str], cols: list[str], stress, results):
    res = []
    prev_time = "2016-01-01 11:11:11"  # arbitary time before the experiment started
    conn, c = get_connection()
    for row in tqdm(stress, total=len(stress)):
        curr_time = row[1]  # 1 means day column -- date
        # query = "SELECT {} FROM sensing WHERE".format(','.join(['coalesce('+x+',-0)' for x in cols]))
        query = "SELECT {} FROM sensing WHERE".format(",".join([x for x in cols]))

        if uids:
            query += " uid IN ({}) AND".format(
                ",".join("'" + str(x) + "'" for x in uids)
            )

        query += " day BETWEEN '{}' AND '{}' ".format(prev_time, curr_time)
        query = query + f" ORDER BY day DESC LIMIT {settings.USE_LAST_N_DAYS_DATA};"
        # print(query)
        c.execute(query)
        r = c.fetchall()
        # print(r)
        res.append([r, row])
        # prev_time = curr_time
    conn.commit()
    results.extend(res)
    # print(results)


def fetch_data_combined(uid, cols, stress):
    res = []
    prev_time = "2016-01-01 11:11:11"  # arbitary time before the experiment started
    conn, c = get_connection()
    for row in tqdm(stress, total=len(stress)):
        curr_time = row[1]  # 1 means day column -- date
        curr_time = datetime.strptime(curr_time, "%Y-%m-%d %H:%M:%S")
        curr_time = curr_time - timedelta(days=1)
        curr_time = curr_time.strftime("%Y-%m-%d %H:%M:%S")
        query = "SELECT day,{} FROM sensing WHERE".format(",".join([x for x in cols]))
        query += ' uid = "{}" AND'.format(uid)
        query += " day BETWEEN '{}' AND '{}' ".format(prev_time, curr_time)
        query = query + f" ORDER BY day DESC LIMIT {settings.USE_LAST_N_DAYS_DATA+4};"
        c.execute(query)
        r = c.fetchall()
        res.append(r)
        prev_time = curr_time

    return res
    # print(results)


def z_score_classificatoin(stresses):
    """
    stress schema : uid, day, stress, mean, std_dev, z_score
    """
    stresses = [list(stress) for stress in stresses]
    for i in range(len(stresses)):
        match settings.CLASS:
            case 5:
                pass
            case 3:
                if stresses[i][5] > 0.5:
                    stresses[i][2] = 2
                elif stresses[i][5] < -0.5:
                    stresses[i][2] = 0
                else:
                    stresses[i][2] = 1
            case 2:
                if stresses[i][5] > 0.5:
                    stresses[i][2] = 2
                elif stresses[i][5] < -0.5:
                    stresses[i][2] = 0
                else:
                    stresses[i][2] = 1
            case _:
                raise ValueError("N-class classification, N value not provided")
    return stresses


def get_counts_of_labels(stresses, ensure_x_levels):
    df = pd.DataFrame(stresses)
    label_counts = df.value_counts().sort_index()
    # df_list = list(label_counts.itertuples(index=False, name=None))
    for i in range(ensure_x_levels):  # This will iterate over 0.0, 1.0, 2.0, and 3.0
        if i not in label_counts:
            print(f"NO {i} label")
            label_counts[i] = 0

    if len(label_counts) != ensure_x_levels:
        raise ValueError("N-class classification, N value not provided")

    return list(label_counts)


def robust_scale_data(data):
    tables = [pd.DataFrame(table) for table in data]
    concatenated_data = pd.concat(tables, ignore_index=True)
    # Step 2: Initialize the RobustScaler and fit it on the concatenated data
    scaler = RobustScaler()
    scaler.fit(concatenated_data)
    # Step 3: Apply the scaler to each table individually
    scaled_tables = [
        pd.DataFrame(scaler.transform(table), columns=table.columns) for table in tables
    ]
    data = [table.values.tolist() for table in scaled_tables]
    return data


from collections import defaultdict


def get_quartile_classification(stresses):
    def get_segmented_three_grouped_series(series, one, two):
        # Sum frequencies in each part
        part1_sum = series.iloc[:one].sum()  # Sum from index 0 to marker1-1
        part2_sum = series.iloc[one:two].sum()  # Sum from marker1 to marker2-1
        part3_sum = series.iloc[two:].sum()
        new_series = pd.Series({0: part1_sum, 1: part2_sum, 2: part3_sum})
        print(new_series)
        return new_series

    def get_segmented_two_grouped_series(series, one):
        # Sum frequencies in each part
        part1_sum = series.iloc[:one].sum()  # Sum from index 0 to marker1-1
        part2_sum = series.iloc[one:].sum()  # Sum from marker1 to marker2-1
        new_series = pd.Series({0: part1_sum, 1: part2_sum})
        # print(new_series)
        return new_series

    # Get the indices of where to place the markers
    def find_min_variance_two_split(series):
        n = len(series)
        best_var = float("inf")
        best_split = None

        for i in range(1, n):
            var = np.var(get_segmented_two_grouped_series(series, i))
            # print(var)
            # print("\n")
            if var < best_var:
                best_var = var
                best_split = i
        return best_split

    def find_min_variance_three_split(series):
        n = len(series)
        best_var = float("inf")
        best_split = None

        for i in range(1, n):
            for j in range(i + 1, n):
                print(i, j)
                var = np.var(get_segmented_three_grouped_series(series, i, j))
                print(var)
                print("\n")
                if var < best_var:
                    best_var = var
                    best_split = (i, j)
        print(best_split)
        return best_split

    grouped_stresses = defaultdict(list)
    bin_new_classed = []
    # Group stresses by the first element (uid) in each subarray
    for stress in stresses:
        grouped_stresses[stress[0]].append(stress)

    for uid, group in grouped_stresses.items():
        # Get the optimal markers
        labels = get_counts_of_labels(group, 5)
        class_freq = pd.Series(labels)
        group = [list(stress) for stress in group]
        marker = (
            find_min_variance_three_split(class_freq)
            if settings.CLASS == 3
            else find_min_variance_two_split(class_freq)
        )
        for i in range(len(group)):
            match settings.CLASS:
                case 5:
                    pass
                case 3:
                    if group[i][2] < marker[0]:
                        group[i][2] = 0
                    elif group[i][2] >= marker[1]:
                        group[i][2] = 2
                    else:
                        group[i][2] = 1
                case 2:
                    if group[i][2] < marker:
                        group[i][2] = 0
                    else:
                        group[i][2] = 1
                case _:
                    raise ValueError("N-class classification, N value not provided")
        bin_new_classed += group
    return bin_new_classed


def melt_data_hourly(data, cols=None):
    if cols is None:
        cols = settings.FEATURES
    if not settings.HOURLY:
        return data
    if len(data[0][0]) != len(cols):
        raise Exception("Feature length mismatch")

    melted_data = []
    for action_sequence in data:
        og = np.array(action_sequence)
        result = []
        # Loop through 0 to 23 hours and extract slices in one loop
        for i in range(0, settings.USE_LAST_N_DAYS_DATA):
            for j in range(0, 24):
                hour_data = og[i][
                    j::24
                ]  # Extract every 24th element starting from index i
                # extend the hour data with the current hour
                result.append(list(reversed(hour_data.tolist())))

        melted_data.append(result)

    return melted_data


def get_low_mediumhigh_classification(stresses):
    stresses = [list(stress) for stress in stresses]
    for i in range(len(stresses)):
        match settings.CLASS:
            case 5:
                pass
            case 3:
                raise ValueError("Not implemented ")
                # if stresses[i][5]>0.5:
                #     stresses[i][2]=2
                # elif stresses[i][5]< -0.5:
                #     stresses[i][2]=0
                # else:
                #     stresses[i][2]=1
            case 2:
                if stresses[i][2] > 3:
                    stresses[i][2] = 1
                elif stresses[i][2] < 2:
                    stresses[i][2] = 0
                else:
                    stresses[i][2] = -1
            case _:
                raise ValueError("N-class classification, N value not provided")
    return stresses


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# def pca_analysis(stress):
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform([x[2:] for x in stress]) # exlcude uid and day

#     # Increase the scale of the column you want to emphasize
#     column_to_emphasize = -1  # Index of the last  column where there is "stress"
#     X_scaled[:, column_to_emphasize] *= 3  # Double its scale to give more importance to "stress"

#     # Then perform PCA on X_scaled


#     pca = PCA(n_components=1)  # Set to 1 for 1D latent space
#     X_pca = pca.fit_transform(X_scaled)
#     print("skew",stats.skew(X_pca))
#     print("mean",int(np.mean(X_pca)))
#     X_pca = stats.zscore(X_pca)
#     # print(X_pca)
#     # X_pca= np.where(X_pca>0.5,2,np.where(X_pca< -0.5,0,1))
#     X_pca= np.where(X_pca>0.5,1,np.where(X_pca< -0.5,0,-1))
#     # X_pca= np.where(X_pca>=0,1,0)
#     new =  [[x[0][0],x[0][1],x[1][0]] for x in zip(stress,X_pca.tolist())] #from stress - uid and day , from xpca - latent value 0 meaning x component in pca
#     return new
def pca_analysis(stress, window=7):  # Add window parameter for rolling statistics
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform([x[2:] for x in stress])  # Exclude uid and day

    # Increase the scale of the column you want to emphasize
    column_to_emphasize = -1  # Index of the last column where there is "stress"
    X_scaled[:, column_to_emphasize] *= 1.5  # Emphasize "stress"

    # Perform PCA on X_scaled
    pca = PCA(n_components=1)  # Set to 1 for 1D latent space
    X_pca = pca.fit_transform(X_scaled)

    print("Skew:", stats.skew(X_pca))
    print("Mean:", int(np.mean(X_pca)))

    # Calculate rolling z-values with min_periods
    df = pd.DataFrame(X_pca, columns=["PCA_Value"])
    rolling_mean = df["PCA_Value"].rolling(window=window, min_periods=1).mean()
    rolling_std = df["PCA_Value"].rolling(window=window, min_periods=1).std()

    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)  # Replace zero std with NaN for safety
    rolling_z = (df["PCA_Value"] - rolling_mean) / rolling_std
    # Fill NaNs in rolling_z
    rolling_z = rolling_z.ffill()
    # rolling_z= np.where(rolling_z>2.5,0,rolling_z)
    # rolling_z= np.where(rolling_z< -2.5,0,rolling_z)
    rolling_z = np.where(rolling_z > 0.3, 1, np.where(rolling_z < -0.3, 0, -1))
    # rolling_z= np.where(rolling_z>0,1,0)
    # rolling_z= np.where(rolling_z>0.5,2,np.where(rolling_z< -0.5,0,1))
    # print(rolling_z.tolist())
    # Assign z-values to the original data
    stress_with_z = [[x[0][0], x[0][1], x[1]] for x in zip(stress, rolling_z.tolist())]

    return stress_with_z


def bin_data(data):
    new_data = []
    for seq in data:
        seq = np.array(seq)
        seq = stats.zscore(seq)
        seq = np.where(
            seq < -1.5,
            0,  # Bin -1 for values < 0
            np.where(
                seq < -0.5,
                1,  # Bin 0 for [0, 1)
                np.where(
                    seq < 0.5,
                    2,  # Bin 1 for [1, 2)
                    np.where(seq < 1.5, 3, 4),  # Bin 2 for [2, 3)   # Bin 3 for [3, 4)
                ),
            ),
        )  # Bin 4 for values >= 4
        new_data.append(seq.tolist())
    return new_data


# def z_score_scale(data):


#     new_data = []
#     for seq in data:
#         seq = np.array(seq,dtype=float)
#         seq = stats.zscore(seq,axis=0)
#         seq = np.nan_to_num(seq, nan=0)
#         scaler = StandardScaler()
#         seq = scaler.fit_transform(seq)
#         new_data.append(seq.tolist())
#     return new_data
def z_score_scale(data):
    # Ensure all elements are converted to floats and handle errors
    try:
        data = [
            (np.array(list(reversed(seq)), dtype=float)) for seq in data
        ]  # <- so that latest data is processed last in lstm and it is more relevant in making prediction. dta is fetched as latest at first. if latest data at first then
    except ValueError as e:
        raise ValueError(f"Data contains non-numeric values: {e}")

    # Concatenate for global statistics calculation
    all_data = np.vstack(data)  # Use vstack to combine along rows

    # Apply Z-score scaling
    z_scaled = stats.zscore(all_data, axis=0, nan_policy="omit")
    z_scaled = np.nan_to_num(
        z_scaled, nan=0
    )  # Handle NaNs and apply 0 meaning it is average

    # z_scaled = np.apply_along_axis(lambda x: np.convolve(x, np.ones(3)/3, mode='same'), axis=0, arr=z_scaled)

    # Apply Standard Scaling
    scaler = StandardScaler()
    standard_scaled = scaler.fit_transform(z_scaled)

    # Split back into original sequence shapes
    split_indices = np.cumsum([len(seq) for seq in data[:-1]])
    scaled_data = np.split(standard_scaled, split_indices)
    return [seq.tolist() for seq in scaled_data]


def add_time_scope(X, timestamps):
    new_data = []
    for x, y in zip(X, timestamps):
        x = pd.DataFrame(x)
        y = pd.to_datetime(y)
        x["day"] = y.dt.dayofweek
        x["day"] = ((2 * x["day"]) / 11) - 1
        x["week"] = y.dt.isocalendar().week
        x["week"] = ((2 * y.dt.isocalendar().week) / 51) - 1
        new_data.append(x.values.tolist())
    return new_data


def gen_daily_seasonal_features(sequences):
    # for each day,generate sin and cos value for each hour, weekdays, and month
    month_and_day_interpolation = []
    # days in 1d array of date strings so need to convert into datetime obj
    for seq in sequences:
        seq_interpolation = []
        for d in seq:
            # weekday value
            d = datetime.strptime(d[0], "%Y-%m-%d %H:%M:%S")
            sin_val_d = np.sin(2 * np.pi * d.weekday() / 7)
            cos_val_d = np.cos(2 * np.pi * d.weekday() / 7)
            sin_val_m = np.sin(2 * np.pi * d.month / 12)
            cos_val_m = np.cos(2 * np.pi * d.month / 12)
            seq_interpolation.append([sin_val_d, cos_val_d, sin_val_m, cos_val_m])
        month_and_day_interpolation.append(seq_interpolation)
    return month_and_day_interpolation


def gen_hourly_seasonal_features(sequences):
    # for each day,generate sin and cos value for each hour, weekdays, and month
    hour_interpolation = []
    # days in 1d array of date strings so need to convert into datetime obj
    for seq in sequences:
        seq_interpolation = []
        for d in seq:
            for j in range(23, -1, -1):
                sin_val_d = np.sin(2 * np.pi * j / 24)
                cos_val_d = np.cos(2 * np.pi * j / 24)
                seq_interpolation.append([sin_val_d, cos_val_d])
        hour_interpolation.append(seq_interpolation)
    return hour_interpolation


def add_daily_seasonal_features(sequences, days):
    for i, seq in enumerate(sequences):
        for j, d in enumerate(seq):
            d.append(days[i][j][0])
            d.append(days[i][j][1])
            d.append(days[i][j][2])
            d.append(days[i][j][3])
    return sequences


def add_hourly_seasonal_features(sequences, days):
    for i, seq in enumerate(sequences):
        for j, d in enumerate(seq):
            d.append(days[i][j][0])
            d.append(days[i][j][1])
    return sequences


def lag_features(sequences):
    seqs = []
    for seq in sequences:
        seq = pd.DataFrame(seq)
        for col in seq.columns:
            seq[f"{col}_lag_1"] = seq[col].rolling(window=3).mean()

            # backward fill the nan values
            seq[f"{col}_lag_1"] = seq[f"{col}_lag_1"].bfill()

        seqs.append(seq.values.tolist())
    return seqs


def median_classification(stress, window=15, evaluate=False):
    # Extract the stress values (exclude uid and day)
    stress_values = np.array([x[-1] for x in stress])
    # Calculate the rolling median for segmentation
    df = pd.DataFrame(stress_values, columns=["Stress_Value"])
    # rolling_median = df["Stress_Value"].rolling(window=window, min_periods=1).median()
    rolling_median = df["Stress_Value"].median()

    # Segregate based on the median
    # Assign 1 for values above the median, 0 for values below, -1 for values close to the median
    # median_value = rolling_median.median()  # Use overall median as the threshold
    median_value = rolling_median
    # segregated = np.where(
    #     stress_values > median_value, 1, np.where(stress_values < median_value, -1, 0)
    # )
    segregated = np.where(
        stress_values > median_value, 1, np.where(stress_values < median_value, 0, 0)
    )
    if evaluate:
        print("NO SJLHERLJGOIFG")
        segregated = np.where(
            stress_values >= median_value,
            1,
            np.where(stress_values < median_value, 0, -1),
        )
    # Assign the segregated values to the original data
    stress_with_segregation = [
        [x[0], x[1], s] for x, s in zip(stress, segregated.tolist())
    ]
    return stress_with_segregation


def mad_scale(data):
    # Ensure all elements are converted to floats and handle errors
    try:
        data = [
            (np.array(list(reversed(seq)), dtype=float)) for seq in data
        ]  # Reversing data so that the latest data comes first
    except ValueError as e:
        raise ValueError(f"Data contains non-numeric values: {e}")

    # Concatenate all sequences for global MAD calculation
    all_data = np.vstack(data)

    # Calculate the median and MAD for scaling
    median = np.median(all_data, axis=0)
    mad = np.median(np.abs(all_data - median), axis=0)

    # Avoid division by zero by replacing 0 MAD values with 1
    mad = np.where(mad == 0, 1, mad)

    # Apply MAD scaling: (data - median) / MAD
    mad_scaled = (all_data - median) / mad

    # Optionally apply Standard Scaling after MAD
    scaler = StandardScaler()
    standard_scaled = scaler.fit_transform(mad_scaled)

    # Split the scaled data back into the original sequence shapes
    split_indices = np.cumsum([len(seq) for seq in data[:-1]])
    scaled_data = np.split(standard_scaled, split_indices)
    return [seq.tolist() for seq in scaled_data]
