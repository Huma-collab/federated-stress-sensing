from tqdm import tqdm
from config import settings
import torch
import os
from .utils import create_path_if_not_exist
import numpy as np


def train(model, optimizer, criterion, writer, epochs):
    model.to(device=settings.DEVICE)
    model.train()
    if model.converged:
        return model.state_dict()
    cumulative_epoch_loss = 0
    total_batches = 0
    # Set patience and min_delta
    patience = settings.PATIENCE
    min_delta = settings.MIN_DELTA
    # Track validation loss
    best_val_loss = float("inf")
    patience_counter = 0
    best_model = model.state_dict()  # Save model's state_dict
    # Initialize the scheduler with the optimizer
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    # get dataloader
    for epoch in (loss_ := tqdm(range(epochs), total=epochs)):
        for batch, (X, y) in enumerate(model.train_dataloader):
            optimizer.zero_grad()
            outputs, _ = model(X)
            loss = criterion(outputs, y.long())
            cumulative_epoch_loss += loss.item()
            total_batches += 1
            loss_.set_postfix_str(f"Loss is at:{loss.item():.4f}")
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=6.0)
            optimizer.step()

        loss = cumulative_epoch_loss / total_batches
        cumulative_epoch_loss = 0
        total_batches = 0
        writer.add_scalar(f"Loss/Train {model.user}", loss, epoch)
        # Val phase
        if model.val_dataloader is not None:
            val_loss = evaluate(model, optimizer, criterion, writer)
            # val_loss = loss
            writer.add_scalar(f"Loss/Val {model.user}", val_loss, epoch)
            # scheduler.step(val_loss)
            # Check for early stopping
            if epoch <= 10:
                patience_counter = 0  # dont concern with patience for first 10 epochs
            if val_loss < best_val_loss:
                best_model = model.state_dict()  # Save model's state_dict

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping")
                    # model.converged = True
                    break
    model.load_state_dict(best_model)
    test_accuracy, _ = accuracy(model, model.val_dataloader, criterion, writer)
    with open(f"logs/{model.user}.txt", "a") as f:
        f.write(f"{test_accuracy}\n")
    writer.flush()
    writer.close()
    path = create_path_if_not_exist(
        f"./saved_models/{model.name}/{settings.NAME}/{model.user}"
    )
    saved_model = model.state_dict()
    torch.save(saved_model, path + ".pth")
    # model.save()
    return saved_model


# returns just the loss
def evaluate(model, optimizer, criterion, writer):
    model.eval()
    cumulative_loss = 0
    total_batches = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(model.val_dataloader):
            outputs, _ = model(X)
            loss = criterion(outputs, y.long())
            cumulative_loss += loss.item()
            total_batches += 1
    # print(f"cum loss: {cumulative_loss} total bat: {total_batches}")
    if total_batches == 0:
        return 0
    avg_loss = cumulative_loss / len(model.val_dataloader)
    model.train()
    return avg_loss


from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def accuracy(model, dataloader, criterion, writer=None):
    model.eval()
    all_preds = []
    all_labels = []
    latent_vectors = []
    loss = 0
    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            outputs, latent = model(X)
            # latent_vectors.append(latent.cpu().numpy())
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            loss += criterion(outputs, y.long())
            # probabilities = softmax(outputs)  # Shape: [batch_size, 2]
            # print("actual",y)
            # print("pred prob",probabilities)
            # print("pred",preds)

    report = classification_report(all_labels, all_preds, digits=2, zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)

    loss /= len(dataloader)
    print(report)
    print(accuracy)
    # print(latent_vectors)
    # latent_vectors = np.concatenate(latent_vectors,axis=0)
    # if writer is not None: writer.add_embedding(latent_vectors,metadata=all_labels)
    return accuracy, loss
