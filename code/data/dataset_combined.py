from torch.utils.data import Dataset, DataLoader, random_split, Subset
import pandas as pd
from datetime import datetime
from .utils import *
from config import settings
import torch
from torch.utils.data import WeightedRandomSampler, SequentialSampler
import threading
import matplotlib.pyplot as plt
import random
from torch.utils.data import Sampler, ConcatDataset

pd.set_option('future.no_silent_downcasting', True)
class Data(Dataset):
    def get_sampler(self, dataset):
        labels = torch.tensor([x[1].cpu().item() for x in dataset], dtype=torch.long)
        label_counts = get_counts_of_labels(
            [x[1].cpu().tolist() for x in dataset], settings.CLASS
        )
        # FIX 3 — clamp to prevent division by zero when a class is missing
        label_counts_tensor = torch.tensor(label_counts, dtype=torch.float)
        if torch.any(label_counts_tensor == 0):
            print(f"WARNING: missing class in sampler, counts: {label_counts}. Clamping.")
            label_counts_tensor = torch.clamp(label_counts_tensor, min=1)
        class_weights = 1.0 / label_counts_tensor
        samples_weights = class_weights[labels]
        samples_weights = samples_weights / samples_weights.sum()
        return WeightedRandomSampler(
            samples_weights, num_samples=len(samples_weights), replacement=False
        )

        # return SequentialSampler(dataset)

    def __init__(self, uid, cols: list[str], evaluate=False) -> None:
        print("-------", uid)
        self.user_id_str = uid[0]
        # FIX 1 — filter before median so threshold is computed on surviving samples only
        raw_stress = fetch_mental_pressure(uid)
        Xy_raw = fetch_data_combined(uid[0], cols, raw_stress)
        Xy_raw = [[x[0], x[1]] for x in zip(Xy_raw, raw_stress)]
        Xy_raw = list(filter(lambda x: len(x[0]) >= settings.USE_LAST_N_DAYS_DATA, Xy_raw))
        Xy_raw = list(map(lambda x: [x[0][0:settings.USE_LAST_N_DAYS_DATA], x[1]], Xy_raw))
        surviving_stress = [a[1] for a in Xy_raw]
        print(evaluate)
        surviving_stress = median_classification(surviving_stress, evaluate=evaluate)
        Xy_filtered = [
            [Xy_raw[i][0], surviving_stress[i]]
            for i in range(len(surviving_stress))
            if surviving_stress[i][2] >= 0
        ]
        just_X = [a[0] for a in Xy_filtered]
        self.stress = [a[1] for a in Xy_filtered]
        self.dates = [x[1] for x in self.stress]
        # Convert to DataFrame with column names from settings
        settings.ALL_HEADINGS_W_HR.insert(0, "day")
        just_X = [pd.DataFrame(x, columns=settings.ALL_HEADINGS_W_HR) for x in just_X]
        print(len(just_X))
        
     

        # ─────────────────────────────────────────────
        # IMPUTATION — happens here, per student, 
        # after DataFrames created, before feature split
        # ─────────────────────────────────────────────

        # Step 1 — chronological split to get train portion
        split_idx = int(0.6 * len(just_X))
        train_dfs = just_X[:split_idx]
        val_dfs   = just_X[split_idx:]

        # Step 2 — compute fill statistics from train only
        # concatenate all train windows to get per-column means
        # change this
        # Step 2 — compute fill statistics from train only
        if len(train_dfs) > 0:
            train_concat = pd.concat(train_dfs, ignore_index=True)
        else:
            train_concat = pd.concat(just_X, ignore_index=True)

        train_means = train_concat.mean(numeric_only=True)

        # Step 3 — define column categories
        # venue still/unlock columns — 100% confirmed structural absence
        # fill with zero
        venue_zero_cols = [
            c for c in train_means.index 
            if any(p in c for p in [
                'loc_social', 'loc_other_dorm',
                'loc_food_still', 'loc_food_unlock',
                'loc_study_still', 'loc_study_unlock',
                'loc_self_dorm', 'loc_home_still',
                'loc_home_unlock'
            ])
        ]

        # venue movement/duration columns — temporal clustering 0.439
        # forward fill max 2 days then zero
        venue_ffill_cols = [
            c for c in train_means.index
            if any(p in c for p in [
                'loc_food_dur', 'loc_social_dur',
                'loc_other_dorm_dur', 'loc_self_dorm_dur',
                'loc_study_dur', 'loc_leisure_dur',
                'loc_worship_dur', 'loc_workout_dur'
            ])
        ]

        # everything else — sensor columns
        # forward fill then backward fill then per-client train mean
        # make sure day column is never touched by imputation
        non_feature_cols = ['day', 'uid']

        sensor_cols = [
            c for c in train_means.index
            if c not in non_feature_cols
            and c not in venue_zero_cols
            and c not in venue_ffill_cols
        ]

        # Step 4 — imputation function
        def impute_window(df):
            df = df.copy()
            
            # venue zero fill
            existing_venue_zero = [c for c in venue_zero_cols if c in df.columns]
            df[existing_venue_zero] = df[existing_venue_zero].fillna(0).infer_objects(copy=False)
            
            # venue movement — ffill 2 days max then zero
            existing_venue_ffill = [c for c in venue_ffill_cols if c in df.columns]
            df[existing_venue_ffill] = (
                df[existing_venue_ffill]
                .ffill(limit=2)
                .fillna(0)
            )
            
            # sensor — ffill, bfill, then per-client train mean
            existing_sensor = [c for c in sensor_cols if c in df.columns]
            df[existing_sensor] = df[existing_sensor].ffill().bfill()
            df[existing_sensor] = df[existing_sensor].fillna(
                train_means[existing_sensor]
            )
            
            return df

        # Step 5 — apply imputation to train and val separately
        train_dfs = [impute_window(df) for df in train_dfs]
        val_dfs   = [impute_window(df) for df in val_dfs]

        # Step 6 — recombine in chronological order
        just_X = train_dfs + val_dfs

        # Step 7 — assert no NaNs remain before z-scoring
        for i, df in enumerate(just_X):
            numeric_df = df.select_dtypes(include=[float, int])
            if numeric_df.isnull().any().any():
                problem_cols = numeric_df.columns[
                    numeric_df.isnull().any()
                ].tolist()
                print(f"WARNING: NaNs remain in window {i} "
                    f"for user {self.user_id_str}: {problem_cols}")

        # ─────────────────────────────────────────────
        # END IMPUTATION
        # ─────────────────────────────────────────────
        # verify imputation worked
        
        # ... rest of your code unchanged
        # Split into separate feature groups based on settings
        self.days = [df[["day"]].values for df in just_X]
        self.hourly_act_features = [
            df[settings.HEADINGS_HOURLY_ACT_HR].values.tolist() for df in just_X
        ]
        self.hourly_loc_features = [
            df[settings.HEADINGS_HOURLY_LOC_HR].values.tolist() for df in just_X
        ]
        self.hourly_unlock_features = [
            df[settings.HEADINGS_HOURLY_UNLOCK_HR].values.tolist() for df in just_X
        ]
        # self.hourly_features = [df[settings.HEADINGS_HOURLY_UNLOCK_HR+settings.HEADINGS_HOURLY_LOC_HR+settings.HEADINGS_HOURLY_ACT_HR].values.tolist() for df in just_X]

        self.daily_loc_mov_features = [
            df[settings.HEADINGS_DAILY_LOC_MOV].values.tolist() for df in just_X
        ]
        self.daily_loc_still_features = [
            df[settings.HEADINGS_DAILY_LOC_STILL].values.tolist() for df in just_X
        ]
        self.daily_unlock_features = [
            df[settings.HEADINGS_DAILY_UNLOCK].values.tolist() for df in just_X
        ]
        self.daily_sleep_features = [
            df[settings.HEADINGS_DAILY_SLEEP].values.tolist() for df in just_X
        ]
        # self.daily_features = [df[settings.HEADINGS_DAILY_SLEEP+settings.HEADINGS_DAILY_LOC_MOV+settings.HEADINGS_DAILY_LOC_STILL+settings.HEADINGS_DAILY_UNLOCK].values.tolist() for df in just_X]

        # Convert back to list format
        self.daily_seasonal_features = gen_daily_seasonal_features(self.days)
        self.hourly_seasonal_features = gen_hourly_seasonal_features(self.days)

        self.hourly_act_features = melt_data_hourly(
            self.hourly_act_features, settings.HEADINGS_HOURLY_ACT_HR
        )
        self.hourly_loc_features = melt_data_hourly(
            self.hourly_loc_features, settings.HEADINGS_HOURLY_LOC_HR
        )
        self.hourly_unlock_features = melt_data_hourly(
            self.hourly_unlock_features, settings.HEADINGS_HOURLY_UNLOCK_HR
        )
        # self.hourly_features =   melt_data_hourly(self.hourly_features,settings.HEADINGS_HOURLY_UNLOCK_HR+settings.HEADINGS_HOURLY_LOC_HR+settings.HEADINGS_HOURLY_ACT_HR)

        self.hourly_act_features = z_score_scale(self.hourly_act_features)
        self.hourly_act_features = lag_features(self.hourly_act_features)
        self.hourly_act_features = add_hourly_seasonal_features(
            self.hourly_act_features, self.hourly_seasonal_features
        )

        self.hourly_loc_features = z_score_scale(self.hourly_loc_features)
        self.hourly_loc_features = lag_features(self.hourly_loc_features)
        self.hourly_loc_features = add_hourly_seasonal_features(
            self.hourly_loc_features, self.hourly_seasonal_features
        )

        self.hourly_unlock_features = z_score_scale(self.hourly_unlock_features)
        self.hourly_unlock_features = lag_features(self.hourly_unlock_features)
        self.hourly_unlock_features = add_hourly_seasonal_features(
            self.hourly_unlock_features, self.hourly_seasonal_features
        )
        # self.hourly_features = z_score_scale(self.hourly_features)
        # self.hourly_features = add_hourly_seasonal_features(self.hourly_features,self.hourly_seasonal_features)

        self.daily_loc_mov_features = z_score_scale(self.daily_loc_mov_features)
        self.daily_loc_mov_features = lag_features(self.daily_loc_mov_features)
        self.daily_loc_mov_features = add_daily_seasonal_features(
            self.daily_loc_mov_features, self.daily_seasonal_features
        )

        self.daily_loc_still_features = z_score_scale(self.daily_loc_still_features)
        self.daily_loc_still_features = lag_features(self.daily_loc_still_features)
        self.daily_loc_still_features = add_daily_seasonal_features(
            self.daily_loc_still_features, self.daily_seasonal_features
        )

        self.daily_unlock_features = z_score_scale(self.daily_unlock_features)
        self.daily_unlock_features = lag_features(self.daily_unlock_features)
        self.daily_unlock_features = add_daily_seasonal_features(
            self.daily_unlock_features, self.daily_seasonal_features
        )

        self.daily_sleep_features = z_score_scale(self.daily_sleep_features)
        self.daily_sleep_features = lag_features(self.daily_sleep_features)
        self.daily_sleep_features = add_daily_seasonal_features(
            self.daily_sleep_features, self.daily_seasonal_features
        )
        # self.daily_features = z_score_scale(self.daily_features)
        # self.daily_features = add_daily_seasonal_features(self.daily_features,self.daily_seasonal_features)

        self.hourly_act_features = torch.tensor(
            self.hourly_act_features, dtype=torch.float
        ).to(device=settings.DEVICE)
        self.hourly_loc_features = torch.tensor(
            self.hourly_loc_features, dtype=torch.float
        ).to(device=settings.DEVICE)
        self.hourly_unlock_features = torch.tensor(
            self.hourly_unlock_features, dtype=torch.float
        ).to(device=settings.DEVICE)

        # self.hourly_features = torch.tensor(self.hourly_features, dtype=torch.float).to(device=settings.DEVICE)

        self.daily_loc_mov_features = torch.tensor(
            self.daily_loc_mov_features, dtype=torch.float
        ).to(device=settings.DEVICE)
        self.daily_loc_still_features = torch.tensor(
            self.daily_loc_still_features, dtype=torch.float
        ).to(device=settings.DEVICE)
        self.daily_unlock_features = torch.tensor(
            self.daily_unlock_features, dtype=torch.float
        ).to(device=settings.DEVICE)
        self.daily_sleep_features = torch.tensor(
            self.daily_sleep_features, dtype=torch.float
        ).to(device=settings.DEVICE)
        # self.daily_features = torch.tensor(self.daily_features, dtype=torch.float).to(device=settings.DEVICE)

        self.len = len(self.stress)
        self.stress = torch.tensor([x[2] for x in self.stress], dtype=torch.float).to(
            device=settings.DEVICE
        )
        # Clear Xy from memory since we've extracted all needed features
        self.Xy = None
        self.just_X = None

    def get_train_test_dataset(self,train, valid, batch_size):
        train_size = int(train * len(self))
        val_size = int(valid * len(self))
        train_indices = list(range(0, train_size))
        val_indices = list(range(train_size, train_size+val_size))
        test_indices = list(range(train_size+val_size, len(self)))

        train_dataset = Subset(self, train_indices)
        val_dataset = Subset(self, val_indices)
        test_dataset = Subset(self, test_indices)
        return train_dataset,val_dataset,test_dataset
    def get_dataloader(self, train, valid, batch_size):
        if train == 1:
            return DataLoader(
                self, batch_size=batch_size, sampler=SequentialSampler(self)
            )

        train_size = int(train * len(self))
        val_size = len(self) - train_size
        # val_size = int(valid * len(self))
        # test_size = len(self) - train_size - val_size

        # train_dataset, val_dataset, test_dataset = random_split(
        #     self, [train_size, val_size, test_size]
        # )
        # train_dataset, val_dataset = random_split(self, [train_size, val_size])
        # Create the indices for each split
        train_indices = list(range(0, train_size))
        val_indices = list(range(train_size, len(self)))
        # test_indices = list(range(train_size + val_size, len(self)))

        # Create Subsets based on these indices
        train_dataset = Subset(self, train_indices)
        val_dataset = Subset(self, val_indices)
        # test_dataset = Subset(self, test_indices)
        return (
            DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=self.get_sampler(train_dataset),
            ),
            DataLoader(val_dataset, batch_size=batch_size),
            # DataLoader(test_dataset, batch_size=20),
            None,  # no testing just use val as test
        )

    def get_encoding_dataloader(self):
        def collate_fn(batch):
            # Unzip the batch into X and y
            X, y = zip(*batch)  # This separates the input features and labels

            # Convert to tensors and move to device
            X = torch.stack([torch.tensor(x, dtype=torch.float) for x in X]).to(
                device=settings.DEVICE
            )

            return X, y

        dataloader = DataLoader(self, batch_size=1)
        return dataloader

    def __getitem__(self, index):
        return [
            [
                self.hourly_act_features[index],
                self.hourly_loc_features[index],
                self.hourly_unlock_features[index],
                self.daily_loc_mov_features[index],
                self.daily_loc_still_features[index],
                self.daily_unlock_features[index],
                self.daily_sleep_features[index],
            ],
            self.stress[index],
        ]
        # return [[self.hourly_features[index],self.daily_features[index]],self.stress[index]]

    def __len__(self):
        return self.len
