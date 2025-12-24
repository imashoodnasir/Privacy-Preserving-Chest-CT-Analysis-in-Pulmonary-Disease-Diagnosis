import argparse
import os
import torch

from src.utils import set_seed, get_device
from src.data import CTNPZDataset, discover_clients
from src.models import SmallCNN2D
from src.train_utils import make_loader, evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train","val","test"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_classes", type=int, default=3)
    ap.add_argument("--slice_stride", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    clients = discover_clients(args.data_root)

    model = SmallCNN2D(num_classes=args.num_classes).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    all_metrics = []
    for c in clients:
        ds = CTNPZDataset(c, split=args.split, augment=False, slice_stride=args.slice_stride)
        loader = make_loader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        m = evaluate(model, loader, device, num_classes=args.num_classes)
        m["client"] = os.path.basename(c)
        all_metrics.append(m)

    # print summary
    print("Per-client metrics:")
    for m in all_metrics:
        print(f"  {m['client']}: acc={m['acc']:.4f}, f1_macro={m['f1_macro']:.4f}, auc_ovr={m['auc_ovr']}")
    mean_acc = sum(m["acc"] for m in all_metrics)/len(all_metrics)
    mean_f1 = sum(m["f1_macro"] for m in all_metrics)/len(all_metrics)
    aucs = [m["auc_ovr"] for m in all_metrics if m["auc_ovr"] is not None]
    mean_auc = None if not aucs else sum(aucs)/len(aucs)
    print(f"Mean: acc={mean_acc:.4f}, f1_macro={mean_f1:.4f}, auc_ovr={mean_auc}")

if __name__ == "__main__":
    main()
