import os

# os.environ["TQDM_DISABLE"] = "1"
import click
from src.utils import get_model_parameters, get_dataloaders
from src.train import train
from src.predict import predict
from src.flwr import flwr_federated_training
from src.central import train_central
from config import settings
import torch
import numpy as np
from data.dataset_combined import Data

SEED = 88
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(SEED)


@click.command()
@click.option(
    "--task",
    prompt="train or encode?",
    help="determine whether you want to train the model or want to generate the encodings for already trained model",
)
# @click.option(
#     "--model_name",
#     prompt="Model Name: ",
#     default='lstm',
#     help="Takes in the name of model you want to train. Valid inputs: lstmae, others to be added later",
# )
@click.option(
    "--user", prompt="user index", help="user index from 0 - 103", default=None
)
def main(task, user):
    model_name = settings.MODEL_NAME
    print(torch.__version__)
    print(settings.DEVICE)
    if task == "train":
        dataloader = get_dataloaders(task)  # train,val,test
        model, optimizer, criterion, writer = get_model_parameters(
            model_name, dataloader
        )
        train(model, optimizer, criterion, writer, settings.EPOCHS)
    elif task == "flwr":
        flwr_federated_training(model_name, task)
    elif task == "evaluate" or task == "evaluateall":
        if user == None:
            raise Exception("Input user id")
        data = Data(
            [settings.USERS[int(user)]],
            settings.ALL_HEADINGS_W_HR,
            evaluate=True if task == "evaluateall" else False,
        )
        dataloader = get_dataloaders(task, data)  # train,val,test
        model, _, criterion, _ = get_model_parameters(model_name, dataloader)
        # Use full dataset for visualization
        from torch.utils.data import DataLoader
        full_dl = DataLoader(data, batch_size=20)
        predict(model, full_dl, criterion, user, data)
    elif task =="central":
        dataloader = train_central()
        model_name = "centralized"
        model, optimizer, criterion, writer = get_model_parameters(
            model_name, dataloader
        )
        train(model, optimizer, criterion, writer, settings.EPOCHS)
if __name__ == "__main__":
    main()
