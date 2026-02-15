import argparse, pickle, zipfile
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from transformers import SegformerForSemanticSegmentation


def list_images(folder: Path):
    exts = ["*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff"]
    out=[]
    for e in exts:
        out += sorted(folder.glob(e))
    return out


def build_train2raw(raw_class_ids):
    return {i: rid for i, rid in enumerate(raw_class_ids)}


def remap_mask_train_to_raw(mask_train: np.ndarray, train2raw: dict) -> np.ndarray:
    out = np.zeros(mask_train.shape, dtype=np.uint16)
    for tid, rid in train2raw.items():
        out[mask_train == tid] = rid
    return out


@torch.no_grad()
def infer_probs(model, img_rgb, size, mean, std, device):
    tfm = A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2()
    ])
    x = tfm(image=img_rgb)["image"].unsqueeze(0).to(device)
    logits = model(pixel_values=x).logits
    probs = torch.softmax(logits, dim=1)
    return probs


@torch.no_grad()
def predict_tta(model, image_bgr, img_size, mean, std, device, scales=(512, 640)):
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    probs_all = []
    for s in scales:
        p1 = infer_probs(model, img_rgb, s, mean, std, device)

        # flip
        tfm = A.Compose([
            A.Resize(s, s),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2()
        ])
        x = tfm(image=img_rgb)["image"].unsqueeze(0).to(device)
        x_f = torch.flip(x, dims=[3])
        p2 = torch.softmax(model(pixel_values=x_f).logits, dim=1)
        p2 = torch.flip(p2, dims=[3])

        probs_all.append((p1 + p2) / 2.0)

    probs = torch.mean(torch.stack(probs_all, dim=0), dim=0)
    pred = torch.argmax(probs, dim=1)[0].cpu().numpy().astype(np.uint8)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--model_pkl", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./pred_masks")
    ap.add_argument("--zip_name", type=str, default="pred_masks.zip")
    ap.add_argument("--scales", type=int, nargs="+", default=[512, 640])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # load bundle
    bundle = pickle.load(open(args.model_pkl, "rb"))
    model_name = bundle["model_name"]
    state_dict = bundle["state_dict"]
    raw_class_ids = bundle["raw_class_ids"]
    mean = bundle["mean"]
    std = bundle["std"]
    num_classes = bundle["num_classes"]

    train2raw = build_train2raw(raw_class_ids)

    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    root = Path(args.data_root)
    test_img_dir = root / "test_public_80/test_public_80/Color_Images"
    assert test_img_dir.exists()

    test_imgs = list_images(test_img_dir)
    print("Test images:", len(test_imgs))

    for p in tqdm(test_imgs, desc="Infer"):
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        pred_train = predict_tta(model, im, bundle["img_size"], mean, std, device, scales=tuple(args.scales))
        pred_raw = remap_mask_train_to_raw(pred_train, train2raw)
        cv2.imwrite(str(out_dir / f"{p.stem}.png"), pred_raw.astype(np.uint16))

    zip_path = Path(args.zip_name)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out_dir.glob("*.png")):
            z.write(f, arcname=f.name)

    print("✅ Saved masks to:", out_dir)
    print("✅ Saved zip:", zip_path)


if __name__ == "__main__":
    main()
