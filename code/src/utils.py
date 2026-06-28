import shutil
import config.settings as settings
from data.dataset_combined import Data

# from data.dataset import Data
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import pandas as pd


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        elif isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        # Ensure the target has the correct shape (flatten to N,)
        target = target.view(-1)

        # Apply sigmoid to the input logits to get probabilities
        input = torch.sigmoid(input)

        # Compute log of the predicted probabilities (log_sigmoid)
        logpt = F.logsigmoid(input)

        # Gather log probabilities corresponding to the true class
        logpt = logpt.gather(1, target.view(-1, 1))  # 1D target, apply to 2D logpt
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())  # pt = exp(log(pt)), probability for true class

        # If alpha is specified, apply it to the loss
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(
                0, target.data.view(-1)
            )  # Use alpha for the correct class
            logpt = logpt * Variable(at)

        # Compute focal loss component
        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def get_writer(model_name):
    log_dir = f"logs/{model_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(comment="", log_dir=log_dir)
    return writer


def get_dataloaders(task, dataset=None):
    if dataset is None:
        dataset = Data(settings.SELECTED_USERS, settings.ALL_HEADINGS_W_HR)
    if task == "train":
        train_dataloader, valid_dataloader, test_dataloader = dataset.get_dataloader(
            0.6, 0.4, settings.BATCH_SIZE
        )
    elif task == "evaluate":
        # dataset = Data([settings.USERS[int(dataset)]], settings.ALL_HEADINGS_W_HR)
        train_dataloader, valid_dataloader, test_dataloader = dataset.get_dataloader(
            0.6, 0.4, 5000
        )  # only train and val have data test is empty. use valid as test # get all the data at once
        return (
            None,
            valid_dataloader,
            None,
            dataset.user_id_str,
        )
    elif task == "evaluateall":
        # dataset = Data([settings.USERS[int(dataset)]], settings.ALL_HEADINGS_W_HR)

        dl = dataset.get_dataloader(
            1, 0, 5000
        )  # only train and val have data test is empty. use valid as test # get all the data at once
        return (
            None,
            dl,
            None,
            dataset.user_id_str,
        )  # just named as valid_dataloader but is test
    else:
        raise ValueError("task must be either train or encode")
    return train_dataloader, valid_dataloader, test_dataloader, dataset.user_id_str


def get_model_parameters(model_name, dataloader):
    train_dataloader, valid_dataloader, test_dataloader, user_id_str = dataloader


    if model_name == "lstm":
        from model.lstm_classifier import LSTM_CL

        model_settings = settings.LSTM_CLASSIFIER
        model = LSTM_CL(
            (train_dataloader, valid_dataloader, test_dataloader), user_id_str
        )
        model.to(settings.DEVICE)
        criterion = torch.nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.3)
        # criterion = FocalLoss(gamma=1, alpha=None, size_average=True)
        optimizer = optim.AdamW(model.parameters(), lr=model_settings["learning_rate"])
    elif model_name == "lstm_3ch":
        from model.lstm_classifier_3ch import LSTM_CL
        model_settings = settings.LSTM_CLASSIFIER
        model = LSTM_CL(
            (train_dataloader, valid_dataloader, test_dataloader), user_id_str
        )
        model.to(settings.DEVICE)
        criterion = torch.nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.3)
        optimizer = optim.AdamW(model.parameters(), lr=model_settings["learning_rate"])
    elif model_name == "lstm_4ch":
        from model.lstm_classifier_4ch import LSTM_CL
        model_settings = settings.LSTM_CLASSIFIER
        model = LSTM_CL(
            (train_dataloader, valid_dataloader, test_dataloader), user_id_str
        )
        model.to(settings.DEVICE)
        criterion = torch.nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.3)
        optimizer = optim.AdamW(model.parameters(), lr=model_settings["learning_rate"])
    elif model_name == "centralized":
        from model.lstm_classifier_centralized import LSTM_CL

        model_settings = settings.LSTM_CLASSIFIER
        model = LSTM_CL(
            (train_dataloader, valid_dataloader, test_dataloader), user_id_str
        )
        model.to(settings.DEVICE)
        weights = torch.tensor([1.0, 2.5]).to(settings.DEVICE)
        criterion = torch.nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.3,weight=weights)
        # criterion = FocalLoss(gamma=1, alpha=None, size_average=True)
        optimizer = optim.AdamW(model.parameters(), lr=model_settings["learning_rate"])
    else:
        raise ValueError("model_name must be registered")
    writer = get_writer(f"{model.name}/{settings.NAME}/{model.user}")
    return model.to(settings.DEVICE), optimizer, criterion, writer


def get_model(model_name):
    if model_name == "cnn_lstm":
        from model.cnn_gru_classifier import CNN_LSTM_ATT_CL

        model = CNN_LSTM_ATT_CL((None, None, None), "global")
        return model.to(settings.DEVICE)

    elif model_name == "lstm":
        from model.lstm_classifier import LSTM_CL

        model = LSTM_CL((None, None, None), "global")
        model.to(settings.DEVICE)
        return model

    else:
        raise ValueError("model_name must be registered")


def create_path_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def save_representation_as_csv(user, feature, stress, representation, filename):
    """Saves the latent representation of feature in csv file
    Args:
        user (int): User Id 0-59
        feature (string): feature name eg: activity
        stress (int): how stressed is the user for a given representation
        representation (string): serialized array storing the representation
        filename (string) : filename to save the model file
    """
    # length of stress and representtation should be same
    assert len(stress) == len(representation)
    stress = [x[0] for x in stress]
    # convert stress and representation as dataframe with custom column names and save as csv
    df_stress = pd.DataFrame(
        stress, columns=["uid", "day", "level", "mean", "std_dev", "z_score"]
    )
    df_stress.drop("mean", axis=1, inplace=True)
    df_stress.drop("std_dev", axis=1, inplace=True)
    df_stress.drop("z_score", axis=1, inplace=True)
    df_stress = df_stress[["day", "level", "uid"]]
    df_stress["heading"] = "stress"
    # df_stress = pd.DataFrame(stress,columns=['timestamp','level','user','heading'])
    df_stress["feature"] = feature
    df_stress["representation"] = [str(r) for r in representation]
    # print(df_stress.head())
    df_stress.to_csv(f"./{filename}/{user}.csv", index=False)
