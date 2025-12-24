import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim

from src.utils import set_seed, get_device, ensure_dir, save_json
from src.data import CTNPZDataset, discover_clients
from src.models import SmallCNN2D
from src.train_utils import make_loader, train_one_epoch, evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="runs/centralized")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_classes", type=int, default=3)
    ap.add_argument("--slice_stride", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    ensure_dir(args.out_dir)

    clients = discover_clients(args.data_root)

    # Pool train/val across clients
    train_sets = [CTNPZDataset(c, split="train", augment=True, slice_stride=args.slice_stride) for c in clients]
    val_sets = [CTNPZDataset(c, split="val", augment=False, slice_stride=args.slice_stride) for c in clients]

    # naive concatenation without copying data
    train_ds = torch.utils.data.ConcatDataset(train_sets)
    val_ds = torch.utils.data.ConcatDataset(val_sets)

    train_loader = make_loader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SmallCNN2D(num_classes=args.num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = -1.0
    history = []
    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        val_m = evaluate(model, val_loader, device, num_classes=args.num_classes)

        monitor = val_m["auc_ovr"] if val_m["auc_ovr"] is not None else val_m["acc"]
        history.append({"epoch": ep, "train_loss": tr_loss, "val": val_m, "monitor": monitor})
        print(f"[Epoch {ep:03d}] train_loss={tr_loss:.4f} val_acc={val_m['acc']:.4f} val_f1={val_m['f1_macro']:.4f}")

        torch.save({"model": model.state_dict(), "args": vars(args)}, os.path.join(args.out_dir, "last.pt"))
        if monitor > best:
            best = monitor
            torch.save({"model": model.state_dict(), "args": vars(args)}, os.path.join(args.out_dir, "best.pt"))

    save_json(os.path.join(args.out_dir, "history.json"), history)
    print(f"Done. Best monitor={best:.4f}. Checkpoints in {args.out_dir}")

if __name__ == "__main__":
    main()
