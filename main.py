import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import trimesh
from torch_geometric.loader import DataLoader
from torch_geometric.nn.pool import knn_graph

from data      import Rotor37Dataset
from model     import DGCNNReg
from visualize import save_multiple_field
# ─── 1. Configuration ───────────────────────────────────────────────────────────
DATA_ROOT  = "/vols/numeca_nfs01/dling/rotor"
STL_FOLDER = os.path.join(DATA_ROOT, "stl")
CSV_FOLDER = os.path.join(DATA_ROOT, "csv")

TRAIN_SPLIT = "train"
TEST_SPLIT  = "test"

SEED       = 42
BATCH_SIZE = 2
NUM_POINTS = 2048
LR         = 1e-4
EPOCHS     = 100
K          = 20        # k‐NN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED)
np.random .seed(SEED)
torch.manual_seed(SEED)

blade_ids = sorted(os.path.splitext(f)[0]
                   for f in os.listdir(STL_FOLDER) if f.endswith('.stl'))

random.shuffle(blade_ids)

n_train   = int(0.8 * len(blade_ids))
train_ids = blade_ids[:n_train]
test_ids  = blade_ids[n_train:]

with open(os.path.join(DATA_ROOT, 'train_files.txt'), 'w') as f:
    f.writelines(b + '\n' for b in train_ids)
with open(os.path.join(DATA_ROOT, 'test_files.txt'), 'w') as f:
    f.writelines(b + '\n' for b in test_ids)

print(f"Split into {len(train_ids)} train / {len(test_ids)} test blades.", flush=True)




# Loading stats.npz
stats = np.load(os.path.join(DATA_ROOT, "field_bc_stats.npz"))
field_mean = torch.tensor(stats["field_mean"], device=DEVICE)  # [4]
field_std  = torch.tensor(stats["field_std"],  device=DEVICE)  # [4]

# Datasets & Loaders
train_ds = Rotor37Dataset(DATA_ROOT, split=TRAIN_SPLIT, num_points=NUM_POINTS)
test_ds  = Rotor37Dataset(DATA_ROOT, split= TEST_SPLIT, num_points=NUM_POINTS)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  pin_memory=True, num_workers = 4, persistent_workers=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                          shuffle=False, pin_memory=True)

# Model, Loss, Optimizer, Scheduler 
model     = DGCNNReg(k=K, output_dim=4).to(DEVICE)
#if torch.cuda.device_count() > 1:
#    model = nn.DataParallel(model)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
mse_loss  = torch.nn.MSELoss()
mae_loss  = torch.nn.L1Loss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)


# Training & Evaluation 
def train():
    model.train()
    tot_mse = tot_mae = 0.0
    for data in train_loader:
        data = data.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()

        pred_n = model(data)                     # normalized preds
        y_n    = data.y                          # normalized targets

        # denormalize back to physical units
        pred = pred_n * field_std + field_mean
        y    = y_n    * field_std + field_mean

        loss = mse_loss(pred, y)
        loss.backward()
        optimizer.step()

        tot_mse += loss.item()
        tot_mae += mae_loss(pred, y).item()

    return tot_mse / len(train_loader), tot_mae / len(train_loader)


@torch.no_grad()
def evaluate():
    model.eval()
    tot_mse = tot_mae = 0.0
    per_field_rel = np.zeros(4)
    batches = 0

    for data in test_loader:
        data = data.to(DEVICE, non_blocking=True)
        pred_n = model(data)
        y_n    = data.y

        pred = pred_n * field_std + field_mean
        y    = y_n    * field_std + field_mean

        tot_mse += mse_loss(pred, y).item()
        tot_mae += mae_loss(pred, y).item()

        # relative MAE per field
        rel = (pred - y).abs() / y.abs().clamp(min=1e-6)
        per_field_rel += rel.mean(dim=0).cpu().numpy()
        batches += 1

    per_field_rel /= batches
    return tot_mse / batches, tot_mae / batches, per_field_rel * 100.0


# Main Training Loop 
best_rel = float("inf")
history  = {"rel": []}

for epoch in range(1, EPOCHS + 1):
    train_mse, train_mae   = train()
    test_mse, test_mae, rf = evaluate()
    scheduler.step(test_mse)

    history["rel"].append(rf)
    if rf.mean() < best_rel:
        best_rel = rf.mean()
        torch.save(model.state_dict(),
                   os.path.join(DATA_ROOT, "best_rel_model.pth"))

    print(f"Epoch {epoch:03d} | "
          f"Test Rel MAE per field: {[f'{x:.2f}%' for x in rf]}",
          flush=True)

print(f"Training complete. Best avg Rel MAE: {best_rel:.2f}%")

# Plotting
rel_hist = np.vstack(history["rel"])  # [EPOCHS,4]
epochs   = np.arange(1, rel_hist.shape[0] + 1)

plt.figure()
fields = ["Density","Entropy","Pressure","Temp"]
for i,name in enumerate(fields):
    plt.plot(epochs, rel_hist[:, i], label=name)
plt.xlabel("Epoch")
plt.ylabel("Relative MAE (%)")
plt.title("Per-Field Relative MAE over Epochs")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(DATA_ROOT, "rel_error_plot.png"))
plt.show()

