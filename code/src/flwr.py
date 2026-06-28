from collections import OrderedDict
from config import settings
from .utils import *
from .train import train, accuracy
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import flwr as fl
import numpy as np
import torch
from .federated_train import CombinedDataLoader
import os
from datetime import datetime

dataloaders = []
global_test_dataloader = []
fed_writer = get_writer("federated")
global epoch
epoch = 0


def load_dataloaders():
    for client in settings.CLIENTS:
        train_dataloader, valid_dataloader, test_dataloader, user_id_str = (
            get_dataloaders("train", Data([client], settings.ALL_HEADINGS_W_HR))
        )
        dataloaders.append(
            (train_dataloader, valid_dataloader, test_dataloader, user_id_str)
        )
        global_test_dataloader.append(valid_dataloader)


def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def evlauation_metrics_aggregation_fn(metrics):
    global epoch
    print(metrics)
    accuracy = np.mean([m[1]["accuracy"] for m in metrics])
    weighted_f1 = np.mean([m[1]["weighted_f1"] for m in metrics])
    fed_writer.add_scalar(f"Accuracy/Eval", accuracy, epoch)
    epoch += 1

    print(
        "< < < ---------------Average accuracy and weighted f1 on global model using test dataset from all clients:-----------------------------> > >",
        accuracy,
        weighted_f1,
    )
    return {"accuracy": accuracy, "weighted_f1": weighted_f1}


# dont need this. no fit metrics at all
# def fit_metrics_aggregation_fn(metrics):
#     print(metrics)
#     loss = np.mean([m[1]["loss"] for m in metrics])
#     acc = np.mean([m[1]["accuracy"] for m in metrics])
#     print(
#         "------------- Average validation accuracy across clients using client's val dataset:",
#         acc,
#     )
#     print(
#         "------------- Average validation loss across clients using client's val dataset:",
#         loss,
#     )
#     return {"loss": loss}


class FlowerClient(fl.client.NumPyClient):

    def __init__(
        self, model, optimizer, criterion, writer, trainloader, valloader, partition_id
    ):
        self.model = model
        self.model.to(settings.DEVICE)
        self.trainloader = trainloader
        self.valloader = valloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.writer = writer
        self.trainloader = trainloader
        self.valloader = valloader
        self.partition_id = partition_id

        self.best_accuracy = 0.0
        # self.best_model_state = None
        self.model_save_path = f"saved_models/flwr/client-{self.partition_id}"
        os.makedirs(self.model_save_path, exist_ok=True)

        # Load best accuracy if exists
        # self.load_best_model()

    # def load_best_model(self):
    #     # Try to find the best accuracy from existing saved models
    #     model_files = os.listdir(self.model_save_path)
    #     if model_files:
    #         # Find the model file with highest accuracy
    #         model_files = [
    #             (f, float(f.split("_")[1].replace(".pth", "")))
    #             for f in model_files
    #             if f.endswith(".pth")
    #         ]
    #         if model_files:
    #             best_file, best_acc = max(model_files, key=lambda x: x[1])
    #             model_path = os.path.join(self.model_save_path, best_file)

    #             # Load only the model weights
    #             try:
    #                 self.best_model_state = torch.load(
    #                     model_path, weights_only=True  # Safer loading
    #                 )
    #                 self.best_accuracy = best_acc  # Get accuracy from filename
    #                 print(
    #                     f"Loaded previous best model with accuracy: {self.best_accuracy:.4f}"
    #                 )
    #             except Exception as e:
    #                 print(f"Error loading model: {e}")
    #                 self.best_model_state = None
    #                 self.best_accuracy = 0.0

    def save_model(self, accuracy):
        # Save model with client ID and accuracy in filename
        model_path = os.path.join(
            self.model_save_path, f"client-{self.partition_id}-acc_{accuracy:.4f}.pth"
        )

        # Save the model
        torch.save(self.model.state_dict(), model_path)

        # Optionally delete older models to save space
        self.cleanup_old_models()

        print(f"Saved model with accuracy {accuracy:.4f} to {model_path}")

    def cleanup_old_models(self, keep_best_n=3):
        """Keep only the N best models to save space"""
        model_files = os.listdir(self.model_save_path)
        if len(model_files) > keep_best_n:
            # Sort models by accuracy (descending)
            models = [
                (f, float(f.split("_")[1].replace(".pth", "")))
                for f in model_files
                if f.endswith(".pth")
            ]
            models.sort(key=lambda x: x[1], reverse=True)

            # Delete older models
            for model_file, _ in models[keep_best_n:]:
                os.remove(os.path.join(self.model_save_path, model_file))

    def get_parameters(self, config):
        print(f"[Client {self.partition_id}] get_parameters")
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        train(
            self.model, self.optimizer, self.criterion, self.writer, settings.EPOCHS
        )  # Use your train function
        # _, loss = accuracy(self.model, self.trainloader, self.criterion, self.writer)
        # if acc > self.best_accuracy:
        #     self.best_accuracy = acc
        #     self.save_model(acc)
        #     print(f"New best accuracy achieved: {acc:.4f}")
        # else:
        #     print(f"No improvement. Current: {acc:.4f}, Best: {self.best_accuracy:.4f}")
        #     self.model.load_state_dict(self.best_model_state)
        #     acc = self.best_accuracy
        # Differential Privacy — clip weights then add Gaussian noise
        if settings.DP_ENABLED:
            params = self.get_parameters(config)
            dp_params = []
            for p in params:
                # Step 1: clip by L2 norm
                l2_norm = np.linalg.norm(p)
                clip_factor = min(1.0, settings.DP_CLIPPING_THRESHOLD / (l2_norm + 1e-8))
                p_clipped = p * clip_factor
                # Step 2: add Gaussian noise
                noise = np.random.normal(0, settings.DP_NOISE_MULTIPLIER * settings.DP_CLIPPING_THRESHOLD, p.shape)
                dp_params.append(p_clipped + noise)
            noisy_params = dp_params
        else:
            noisy_params = self.get_parameters(config)

        return (
            noisy_params,
            len(self.trainloader),
            {
                "loss": float(1),
                "fit_from": self.partition_id,
            },
        )

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        all_preds = []
        all_labels = []
        loss = 0
        with torch.no_grad():
            for batch, (X, y) in enumerate(self.valloader):

                outputs, latent = self.model(X)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                loss += self.criterion(outputs, y.long())

        loss /= len(self.valloader.dataset)
        # report = classification_report(all_labels, all_preds, digits=2, zero_division=0)
        # print(report)
        accuracy = accuracy_score(all_labels, all_preds)
        # Calculate accuracy score
        print(f"Accuracy: {accuracy:.2f}")

        # Calculate Weighted Average F1 Score, Precision, and Recall
        weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
        f1 = f1_score(all_labels, all_preds, average=None)

        # print("weigh 0 ", f1[0])
        # print("weigh 1 ", f1[1])
        # print(f"Weighted F1-Score: {weighted_f1:.2f}")
        if len(f1) < 2 or f1[0] < 0.1 or f1[1] < 0.1:
            pass  # Not participating
        if weighted_f1 >= self.best_accuracy:
            self.save_model(weighted_f1)
            self.best_accuracy = weighted_f1  # print(report)
        return (
            float(loss),
            len(self.valloader),
            {
                "accuracy": float(accuracy),
                "weighted_f1": float(weighted_f1),
                "eval_from": self.partition_id,
                "loss": float(loss),
            },
        )


def create_client_fn(model_name, dataloaders):
    def client_fn(context: fl.common.Context) -> fl.client.Client:
        client_id = context.node_config["partition-id"]
        (train_dataloader, valid_dataloader, test_dataloader, user_id_str) = (
            dataloaders[client_id]
        )

        model, optimizer, criterion, writer = get_model_parameters(
            model_name,
            (train_dataloader, valid_dataloader, test_dataloader, user_id_str),
        )
        return FlowerClient(
            model,
            optimizer,
            criterion,
            writer,
            train_dataloader,
            valid_dataloader,
            client_id,
        ).to_client()

    return client_fn


# Evaluation function in the end of training
# def get_evaluate(model_name, global_test_dataloader):
#     def evaluate(server_round: int,parameters,config):
#         #change this dataloader to another user's dataloader for loso kind of evaluation
#         model, optimizer, criterion, writer = get_model_parameters(model_name,(None,global_test_dataloader,None,'flwrglobal') )
#         set_parameters(model, parameters)  # Update model with the latest parameters
#         acc, loss = accuracy(model,global_test_dataloader,criterion)
#         # print(f"Server-side evaluation loss {loss} / accuracy {acc}")
#         print("------------- Average accuracy across clients using client's test dataset:",acc)
#         return loss, {"accuracy": acc}
#     return evaluate


def get_server_fn(model_name, global_test_dataloader):
    def server_fn(context: fl.common.Context) -> fl.server.ServerAppComponents:
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1,
            min_fit_clients=len(settings.CLIENTS),
            min_evaluate_clients=2,
            min_available_clients=len(settings.CLIENTS),
            evaluate_metrics_aggregation_fn=evlauation_metrics_aggregation_fn,
            accept_failures=True,
        )
        model, _, _, _ = get_model_parameters(model_name, (None, None, None, "global"))

        # server can only do aggregation on client's accuracy
        # if server wants to do validation on its own, it has to use client that is not used in the training
        #     strategy = fl.server.strategy.FedProx(
        #         fraction_fit=1.0,
        #         fraction_evaluate=1,
        #         min_fit_clients=len(settings.CLIENTS),
        #         min_evaluate_clients=2,
        #         min_available_clients=len(settings.CLIENTS),
        #         initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(model)),
        #         fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, # aggregates all the accuracy from the client while training aftereach run
        #         # evaluate_fn=get_evaluate(model_name,global_test_dataloader), # CANT DO! PRIVACY ISSUES! server side validation
        #         evaluate_metrics_aggregation_fn=evlauation_metrics_aggregation_fn,
        #         accept_failures=False,
        #         proximal_mu=0.1
        # )
        config = fl.server.ServerConfig(num_rounds=settings.GLOBAL_EPOCHS)
        return fl.server.ServerAppComponents(strategy=strategy, config=config)

    return server_fn


def flwr_federated_training(model_name, task):
    load_dataloaders()
    cdl = CombinedDataLoader(global_test_dataloader)

    server = fl.server.ServerApp(server_fn=get_server_fn(model_name, cdl))

    client_fn_with_model = create_client_fn(model_name, dataloaders)
    # remove saved_model/flwr folder to removed any saved model from previous runs
    if os.path.exists("saved_models/flwr"):
        shutil.rmtree("saved_models/flwr")
    # Start the Flower server
    fl.simulation.run_simulation(
        server_app=server,
        client_app=fl.client.ClientApp(client_fn=client_fn_with_model),
        num_supernodes=len(settings.CLIENTS),  # Number of clients
        verbose_logging=False,
        backend_config={
            "client_resources": {"num_cpus": 2, "num_gpus": 0.25}
        },  # Adjust based on your setup
    )
