import argparse
import os
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from src.utils import set_seed, get_device, ensure_dir, save_json, model_to_cpu_state
from src.data import CTNPZDataset, discover_clients
from src.models import SmallCNN2D
from src.train_utils import make_loader, train_one_epoch, evaluate
from src.federated import fedavg_aggregate, select_clients

def run(args):
    set_seed(args.seed)
    device = get_device(args.device)

    clients = discover_clients(args.data_root)
    ensure_dir(args.out_dir)

    # infer classes by scanning a few files (toy sets always have labels 0..C-1)
    # here we simply take args.num_classes; for real data, set it explicitly.
    num_classes = args.num_classes

    global_model = SmallCNN2D(num_classes=num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()

    best_metric = -1.0
    history = []

    for rnd in range(1, args.rounds + 1):
        round_seed = args.seed + rnd
        selected = select_clients(clients, args.clients_per_round, seed=round_seed)

        client_states = []
        client_sizes = []
        client_logs = []

        global_state_cpu = model_to_cpu_state(global_model)

        for client_path in selected:
            # local data
            train_ds = CTNPZDataset(client_path, split="train", augment=True, slice_stride=args.slice_stride)
            val_ds = CTNPZDataset(client_path, split="val", augment=False, slice_stride=args.slice_stride)
            train_loader = make_loader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            val_loader = make_loader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

            # local model clone
            local_model = SmallCNN2D(num_classes=num_classes).to(device)
            local_model.load_state_dict({k: v.to(device) for k, v in global_state_cpu.items()})

            optimizer = optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            # local training
            for _ in range(args.local_epochs):
                tr_loss = train_one_epoch(local_model, train_loader, optimizer, device, loss_fn)

            # local validation
            val_metrics = evaluate(local_model, val_loader, device, num_classes=num_classes)

            # collect
            st_cpu = model_to_cpu_state(local_model)
            client_states.append(st_cpu)
            client_sizes.append(len(train_ds))
            client_logs.append({"client": os.path.basename(client_path), "val": val_metrics})

        # aggregate
        new_global_state = fedavg_aggregate(
            global_state=global_state_cpu,
            client_states=client_states,
            client_sizes=client_sizes,
            dp_sigma=args.dp_sigma,
            dp_clip=args.dp_clip,
            seed=args.seed + 999 * rnd,
        )
        global_model.load_state_dict({k: v.to(device) for k, v in new_global_state.items()})

        # evaluate global model on each client's validation set
        per_client_val = []
        for client_path in clients:
            val_ds = CTNPZDataset(client_path, split="val", augment=False, slice_stride=args.slice_stride)
            val_loader = make_loader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            m = evaluate(global_model, val_loader, device, num_classes=num_classes)
            per_client_val.append({"client": os.path.basename(client_path), **m})

        # pick a single monitoring metric: mean AUC if available else mean acc
        aucs = [x["auc_ovr"] for x in per_client_val if x["auc_ovr"] is not None]
        if len(aucs) > 0:
            monitor = float(sum(aucs) / len(aucs))
            monitor_name = "mean_auc_ovr"
        else:
            monitor = float(sum(x["acc"] for x in per_client_val) / len(per_client_val))
            monitor_name = "mean_acc"

        rec = {
            "round": rnd,
            "selected_clients": [os.path.basename(p) for p in selected],
            "client_logs": client_logs,
            "global_val": per_client_val,
            "monitor_name": monitor_name,
            "monitor_value": monitor,
        }
        history.append(rec)

        print(f"[Round {rnd:03d}] {monitor_name}={monitor:.4f}")

        # checkpoint
        ckpt_path = os.path.join(args.out_dir, "last.pt")
        torch.save({"model": global_model.state_dict(), "args": vars(args)}, ckpt_path)

        if monitor > best_metric:
            best_metric = monitor
            best_path = os.path.join(args.out_dir, "best.pt")
            torch.save({"model": global_model.state_dict(), "args": vars(args)}, best_path)

    save_json(os.path.join(args.out_dir, "history.json"), history)
    print(f"Done. Best monitor={best_metric:.4f}. Checkpoints in {args.out_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Path to data/ containing client_XX folders")
    ap.add_argument("--out_dir", type=str, default="runs/federated")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--clients_per_round", type=int, default=3)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_classes", type=int, default=3)
    ap.add_argument("--slice_stride", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--dp_sigma", type=float, default=0.0, help="Gaussian noise std for DP-style update (didactic)")
    ap.add_argument("--dp_clip", type=float, default=0.0, help="L2 clip norm for DP-style update (didactic)")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    run(args)

if __name__ == "__main__":
    main()
