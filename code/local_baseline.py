import os
import sys
import numpy as np
import torch
from config import settings
from data.dataset_combined import Data
from src.utils import get_dataloaders, get_model_parameters
from src.train import train, accuracy
from sklearn.metrics import f1_score
import json

# Set seeds for reproducibility
SEED = 88
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

results = {}
all_accuracies = []
skipped_users = []

total_users = len(settings.USERS)
print(f"Starting local-only baseline training for {total_users} users...")
print(f"Each user trains independently with NO weight sharing")
print(f"Settings: {settings.EPOCHS} epochs, batch size {settings.BATCH_SIZE}")
print("="*60)

for i, user in enumerate(settings.USERS):
    try:
        print(f"\n[{i+1}/{total_users}] Training user: {user[:8]}...")
        
        # Create dataset for this user only
        dataset = Data([user], settings.ALL_HEADINGS_W_HR)
        
        # Get train and validation dataloaders
        train_dl, val_dl, _ = dataset.get_dataloader(
            0.6, 0.4, settings.BATCH_SIZE
        )
        
        # Skip if not enough data
        if train_dl is None or val_dl is None:
            print(f"  Skipping - no dataloader created")
            skipped_users.append(user)
            continue
            
        if len(train_dl) == 0:
            print(f"  Skipping - empty training data")
            skipped_users.append(user)
            continue

        # Create model, optimizer, criterion for this user
        # NO global weights - starts from scratch
        dataloader = (train_dl, val_dl, None, user)
        model, optimizer, criterion, writer = get_model_parameters(
            "lstm", dataloader
        )
        
        # Train locally - NO federation, NO weight sharing
        train(model, optimizer, criterion, writer, settings.EPOCHS)
        
        # Evaluate on validation data
        acc, loss = accuracy(model, val_dl, criterion)
        model.eval()
        _preds, _labels = [], []
        with torch.no_grad():
            for X, y in val_dl:
                out, _ = model(X)
                _preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                _labels.extend(y.cpu().numpy())
        wf1 = float(f1_score(_labels, _preds, average="weighted", zero_division=0))
        
        # Save result
        results[user] = {
            "accuracy": float(acc),
            "weighted_f1": wf1,
            "loss": float(loss)
        }
        all_accuracies.append(float(acc))
        
        print(f"  Accuracy: {acc:.4f} | Weighted F1: {wf1:.4f} | Loss: {loss:.4f}")
        
        # Save progress after each user in case of crash
        with open("local_baseline_results.json", "w") as f:
            json.dump({
                "completed_users": len(all_accuracies),
                "skipped_users": len(skipped_users),
                "average_accuracy": float(np.mean(all_accuracies)),
                "per_user": results
            }, f, indent=2)

    except Exception as e:
        print(f"  Error for user {user[:8]}: {e}")
        skipped_users.append(user)
        continue

# Final results
avg = float(np.mean(all_accuracies)) if all_accuracies else 0
min_acc = float(min(all_accuracies)) if all_accuracies else 0
max_acc = float(max(all_accuracies)) if all_accuracies else 0

print("\n" + "="*60)
print("LOCAL-ONLY BASELINE FINAL RESULTS")
print("="*60)
print(f"Total users:              {total_users}")
print(f"Successfully trained:     {len(all_accuracies)}")
print(f"Skipped:                  {len(skipped_users)}")
print(f"Average accuracy:         {avg:.4f}")
print(f"Min accuracy:             {min_acc:.4f}")
print(f"Max accuracy:             {max_acc:.4f}")
print(f"Std deviation:            {float(np.std(all_accuracies)):.4f}")
print("="*60)

# Save final results
final_results = {
    "method": "local_only_baseline",
    "description": "Each client trains independently with no weight sharing",
    "total_users": total_users,
    "trained_users": len(all_accuracies),
    "skipped_users": len(skipped_users),
    "average_accuracy": avg,
    "min_accuracy": min_acc,
    "max_accuracy": max_acc,
    "std_deviation": float(np.std(all_accuracies)),
    "per_user": results
}

with open("local_baseline_results.json", "w") as f:
    json.dump(final_results, f, indent=2)

print("Results saved to local_baseline_results.json")
