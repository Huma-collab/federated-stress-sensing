import os

from src.utils import get_model_parameters, get_dataloaders
from src.train import train
from config import settings
import torch
import numpy as np
from data.dataset_combined import Data
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset, DataLoader,WeightedRandomSampler
from data import utils
def get_sampler(dataset):
    labels = torch.tensor([x[1].cpu().item() for x in dataset], dtype=torch.long)
    label_counts = utils.get_counts_of_labels(
        [x[1].cpu().tolist() for x in dataset], settings.CLASS
    )
    class_weights = 1.0 / torch.tensor(label_counts, dtype=torch.float)
    samples_weights = class_weights[labels]
    samples_weights = samples_weights / samples_weights.sum()
    return WeightedRandomSampler(
        samples_weights, num_samples=len(samples_weights), replacement=False
    )
def train_central():
    # fetch all dataset
    all_train_ds = []
    all_val_ds = []
    all_test_ds = []
    count = 0
    for user in settings.USERS:
        dataset = Data([user], settings.ALL_HEADINGS_W_HR)
        train_ds, val_ds, test_ds = dataset.get_train_test_dataset(0.6, 0.2, settings.BATCH_SIZE)
        all_train_ds.append(train_ds)
        all_val_ds.append(val_ds)
        all_test_ds.append(test_ds)
        # if count == 2:
        #     break;
        count+=1
    all_train_ds = ConcatDataset(all_train_ds)
    all_val_ds = ConcatDataset(all_val_ds)
    all_test_ds = ConcatDataset(all_test_ds)

    
    # create dataloader 
    dataloader=  (
            DataLoader(
                all_train_ds,
                batch_size=settings.BATCH_SIZE,
                sampler=get_sampler(all_train_ds),
            ),
            DataLoader(all_val_ds, batch_size=settings.BATCH_SIZE),
            # DataLoader(test_dataset, batch_size=20),
            DataLoader(all_test_ds, batch_size=settings.BATCH_SIZE),
            "central_low"
        )
    return dataloader
    # model, optimizer, criterion, writer = get_model_parameters(
    #     model_name, dataloader
    # )
    # train(model, optimizer, criterion, writer, settings.EPOCHS)