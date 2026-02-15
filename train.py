import os, random, time, gc, argparse, pickle
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from transformers import SegformerForSemanticSegmentation


# -------------------------
# Utils
# -------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def list_images(folder: Path):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    out = []
    for e in exts:
        out += sorted(folder.glob(e))
    return out


def make_pairs(img_dir: Path, msk_dir: Path):
    imgs = list_images(img_dir)
    msks = list_images(msk_dir)
    img_map = {p.stem: p for p in imgs}
    msk_map = {p.stem: p for p in msks}
    common = sorted(set(img_map.keys()) & set(msk_map.keys()))
    return [(img_map[s], msk_map[s]) for s in common]


# -------------------------
# Label mapping
# -------------------------
RAW_CLASS_IDS = [100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000]
NUM_CLASSES = len(RAW_CLASS_IDS)
RAW2TRAIN = {rid: i for i, rid in enumerate(RAW_CLASS_IDS)}
TRAIN2RAW = {i: rid for i, rid in enumerate(RAW_CLASS_IDS)}
IGNORE_INDEX = 255

def remap_mask_raw_to_train(mask_raw: np.ndarray) -> np.ndarray:
    out = np.full(mask_raw.shape, IGNORE_INDEX, dtype=np.uint8)
    for rid, tid in RAW2TRAIN.items():
        out[mask_raw == rid] = tid
    return out


# -------------------------
# Dataset
# -------------------------
class SegDataset(Dataset):
    def __init__(self, pairs, tfms=None):
        self.pairs = pairs
        self.t = tfms

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]

        img = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        m_raw = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if m_raw.ndim == 3:
            m_raw = cv2.cvtColor(m_raw, cv2.COLOR_BGR2GRAY)
        mask = remap_mask_raw_to_train(m_raw.astype(np.uint16)).astype(np.uint8)

        if self.t:
            aug = self.t(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]

        return img, torch.as_tensor(mask, dtype=torch.long)


# -------------------------
# Losses / metrics
# -------------------------
class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        target_oh = F.one_hot(target.clamp(0, self.num_classes-1), num_classes=self.num_classes).permute(0,3,1,2).float()
        dims = (0,2,3)
        inter = torch.sum(probs * target_oh, dims)
        union = torch.sum(probs + target_oh, dims)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


@torch.no_grad()
def mean_iou(logits, target, num_classes, ignore_index=255):
    pred = torch.argmax(logits, dim=1)
    valid = (target != ignore_index)
    pred = pred[valid]
    target = target[valid]
    ious = []
    for c in range(num_classes):
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union == 0:
            continue
        ious.append(inter / union)
    return float(np.mean(ious)) if len(ious) else 0.0


def compute_class_weights(pairs, max_masks=400):
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    use = pairs[:min(max_masks, len(pairs))]
    for _, mp in tqdm(use, desc="Class weight scan"):
        m_raw = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if m_raw.ndim == 3:
            m_raw = cv2.cvtColor(m_raw, cv2.COLOR_BGR2GRAY)
        m = remap_mask_raw_to_train(m_raw.astype(np.uint16))
        m = m[m != IGNORE_INDEX]
        counts += np.bincount(m.flatten(), minlength=NUM_CLASSES)
    counts = np.maximum(counts, 1.0)
    freq = counts / counts.sum()
    w = 1.0 / np.log(1.02 + freq)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def poly_lr(epoch, max_epochs, base_lr, power=0.9):
    return base_lr * (1 - epoch / max_epochs) ** power


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="./outputs")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Paths
    root = Path(args.data_root)
    train_img = root / "Training/Offroad_Segmentation_Training_Dataset/train/Color_Images"
    train_msk = root / "Training/Offroad_Segmentation_Training_Dataset/train/Segmentation"
    val_img   = root / "Training/Offroad_Segmentation_Training_Dataset/val/Color_Images"
    val_msk   = root / "Training/Offroad_Segmentation_Training_Dataset/val/Segmentation"

    assert train_img.exists() and train_msk.exists()
    assert val_img.exists() and val_msk.exists()

    train_pairs = make_pairs(train_img, train_msk)
    val_pairs   = make_pairs(val_img, val_msk)

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    train_tfms = A.Compose([
        A.Resize(args.img_size, args.img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.20, rotate_limit=15,
                           border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.6),
        A.RandomBrightnessContrast(p=0.6),
        A.HueSaturationValue(p=0.35),
        A.GaussNoise(p=0.20),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ])

    val_tfms = A.Compose([
        A.Resize(args.img_size, args.img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ])

    train_ds = SegDataset(train_pairs, train_tfms)
    val_ds   = SegDataset(val_pairs, val_tfms)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Model
    MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"
    id2label = {i: f"class_{i}" for i in range(NUM_CLASSES)}
    label2id = {v: k for k, v in id2label.items()}

    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    ).to(device)

    # Losses
    class_weights = compute_class_weights(train_pairs).to(device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
    dice_loss = DiceLoss(NUM_CLASSES)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    best_iou = -1.0
    best_ckpt = out_dir / "best_segformer_b2.pt"

    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        lr_now = poly_lr(epoch-1, args.epochs, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        # Train
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                logits = model(pixel_values=x).logits
                logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
                loss = ce_loss(logits, y) + dice_loss(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        # Valid
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [valid]", leave=False):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = model(pixel_values=x).logits
                logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)

                loss = ce_loss(logits, y) + dice_loss(logits, y)
                iou = mean_iou(logits, y, NUM_CLASSES, ignore_index=IGNORE_INDEX)

                val_loss += loss.item()
                val_iou += iou

        val_loss /= max(1, len(val_loader))
        val_iou /= max(1, len(val_loader))

        elapsed = time.time() - t0
        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_mIoU={val_iou:.4f} | lr={lr_now:.2e} | time={elapsed/60:.1f}m")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_miou": val_iou, "lr": lr_now, "time_sec": elapsed})

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "best_iou": best_iou,
                "num_classes": NUM_CLASSES,
                "img_size": args.img_size,
                "model_name": MODEL_NAME
            }, best_ckpt)
            print(f"✅ Saved BEST: mIoU={best_iou:.4f} (epoch {epoch})")

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # Save history CSV
    import pandas as pd
    pd.DataFrame(history).to_csv(out_dir / "history_segformer_b2.csv", index=False)

    # Save model.pkl bundle for submission
    ckpt = torch.load(best_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    bundle = {
        "model_name": ckpt["model_name"],
        "state_dict": model.state_dict(),
        "raw_class_ids": RAW_CLASS_IDS,
        "img_size": ckpt["img_size"],
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "num_classes": NUM_CLASSES
    }
    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump(bundle, f)

    print("✅ Saved:", best_ckpt)
    print("✅ Saved:", out_dir / "model.pkl")


if __name__ == "__main__":
    main()
