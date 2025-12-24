import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np

def train_one_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    losses = []
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0

@torch.no_grad()
def evaluate(model, loader, device, num_classes: int):
    model.eval()
    all_y = []
    all_p = []
    for x, y in tqdm(loader, leave=False):
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_p.append(probs)
        all_y.append(y.numpy())
    y_true = np.concatenate(all_y) if all_y else np.array([])
    p = np.concatenate(all_p) if all_p else np.array([])
    if y_true.size == 0:
        return {"acc": 0.0, "f1_macro": 0.0, "auc_ovr": None}
    y_pred = p.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    auc = None
    try:
        if num_classes == 2:
            auc = roc_auc_score(y_true, p[:,1])
        else:
            auc = roc_auc_score(y_true, p, multi_class="ovr")
    except Exception:
        auc = None
    return {"acc": float(acc), "f1_macro": float(f1m), "auc_ovr": None if auc is None else float(auc)}

def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int = 0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False)
