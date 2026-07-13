"""
CKA 사전검증 — CNN(YOLOv8)과 Transformer(RT-DETR) feature의 "공유 구조" 측정

목적 (ICLR 방향 검증):
  출력 gradient는 직교(cos≈0)였다. 그렇다면 중간 feature 수준에는 두 아키텍처가
  공유하는 구조가 있는가? 있다면 "공유 부분공간 공격"의 근거가 된다.

방법:
  - 두 검출기 백본의 여러 층에 hook을 걸어 feature map 추출.
  - 같은 이미지 배치에 대한 두 모델의 feature를 CKA(Centered Kernel Alignment)로 비교.
    CKA ≈ 1 : 매우 유사(공유 큼) / CKA ≈ 0 : 무관.
  - 층 조합별 CKA 행렬을 만들어 "어느 층끼리 공유가 큰지" 확인.

참고:
  - CKA는 채널 수·해상도가 달라도 비교 가능 (샘플 기준 유사도).
  - gradient 직교(출력)와 대비: feature는 공유가 크게 나오면 → 가설 성립.

사용 예:
  python 05_cka_analysis.py --n 200
  python 05_cka_analysis.py --n 500 --layers 5
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

IMG_SIZE = 640
DATA_FILE = Path("data/person_annotations.json")
OUT_DIR = Path("output")


# ── 데이터 ────────────────────────────────────────────────────────────────────

def load_and_resize(file_name: str, size: int) -> torch.Tensor:
    img = Image.open(Path("data/val2017") / file_name).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


# ── 모델 + hook ───────────────────────────────────────────────────────────────

def load_model(name, device, train_mode=False):
    print(f"  모델 로드: {name}")
    m = YOLO(name)
    nn = m.model.to(device)
    nn.train() if train_mode else nn.eval()
    for p in nn.parameters():
        p.requires_grad_(False)
    return nn


def register_hooks(nn_model, n_layers):
    """중간~후반 Conv2d 중 n_layers개를 균등 선택해 hook (04번과 동일 규칙)."""
    convs = [m for m in nn_model.modules() if isinstance(m, torch.nn.Conv2d)]
    start = int(len(convs) * 0.4)
    pool = convs[start:]
    idxs = [int(i) for i in np.linspace(0, len(pool) - 1, n_layers)]
    chosen = [pool[i] for i in idxs]
    buf = []
    handles = [c.register_forward_hook(lambda m, i, o: buf.append(o)) for c in chosen]
    return buf, handles


def feats_to_matrix(feat_list):
    """[B,C,H,W] feature들을 각각 [B, C*H*W] 벡터로 펴서 반환 (샘플=행)."""
    out = []
    for f in feat_list:
        b = f.shape[0]
        out.append(f.reshape(b, -1).float().cpu())
    return out


# ── CKA (linear) ──────────────────────────────────────────────────────────────

def linear_cka(X, Y):
    """
    선형 CKA. X:[N, d1], Y:[N, d2]  (N=샘플 수).
    반환: 0~1 스칼라 (1=매우 유사).
    HSIC 기반, 특징 차원이 달라도 됨.

    Gram 행렬([N,N]) 방식으로 계산 — feature 차원 d가 수십만이어도
    [d,d] 행렬을 만들지 않아 메모리 안전. ||X^T Y||_F^2 = <XX^T, YY^T> 항등식
    이용, 값은 [d,d] 방식과 동일하나 비용은 O(d^2) → O(N^2).
    """
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    Kx = X @ X.t()   # [N, N]
    Ky = Y @ Y.t()   # [N, N]
    hsic_xy = (Kx * Ky).sum()
    hsic_xx = (Kx * Kx).sum().sqrt()
    hsic_yy = (Ky * Ky).sum().sqrt()
    return (hsic_xy / (hsic_xx * hsic_yy + 1e-12)).item()


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"장치: {device}")

    with open(DATA_FILE) as f:
        records = json.load(f)
    random.shuffle(records)
    records = records[:args.n]
    print(f"분석 이미지: {len(records)}장")

    yolo = load_model("yolov8n.pt", device)
    detr = load_model("rtdetr-l.pt", device, train_mode=True)
    yb, yh = register_hooks(yolo, args.layers)
    db, dh = register_hooks(detr, args.layers)
    print(f"  hook 층 수: YOLO {args.layers} / RT-DETR {args.layers}")

    # 배치로 feature 수집 (샘플 축으로 누적)
    Y_layers = [[] for _ in range(args.layers)]  # YOLO 층별 feature 벡터 누적
    D_layers = [[] for _ in range(args.layers)]  # RT-DETR 층별

    bs = args.batch
    batches = [records[i:i + bs] for i in range(0, len(records), bs)]
    for batch in tqdm(batches, desc="feature 수집"):
        imgs = torch.stack([load_and_resize(r["file_name"], IMG_SIZE) for r in batch]).to(device)
        yb.clear(); db.clear()
        with torch.no_grad():
            yolo(imgs); detr(imgs)
        ym = feats_to_matrix(list(yb))
        dm = feats_to_matrix(list(db))
        for k in range(args.layers):
            Y_layers[k].append(ym[k])
            D_layers[k].append(dm[k])

    for h in yh + dh:
        h.remove()

    Y = [torch.cat(v, 0) for v in Y_layers]   # 각 [N, dim]
    D = [torch.cat(v, 0) for v in D_layers]

    # CKA 행렬: YOLO 층 i vs RT-DETR 층 j
    print("\nCKA 계산 중 (YOLO 층 × RT-DETR 층)...")
    M = np.zeros((args.layers, args.layers))
    for i in range(args.layers):
        for j in range(args.layers):
            M[i, j] = linear_cka(Y[i], D[j])

    # 결과 출력
    print("\n=== CKA 행렬 (행=YOLO 층, 열=RT-DETR 층) ===")
    print("      " + "  ".join([f"D{j}" for j in range(args.layers)]))
    for i in range(args.layers):
        print(f"  Y{i}  " + "  ".join([f"{M[i,j]:.3f}" for j in range(args.layers)]))
    diag = [M[i, i] for i in range(args.layers)]
    print(f"\n대각(같은 깊이) 평균 CKA: {np.mean(diag):.3f}")
    print(f"전체 최대 CKA: {M.max():.3f} / 전체 평균: {M.mean():.3f}")
    print("\n해석:")
    print("  - CKA가 높게(예: >0.3~0.5) 나오면 → 두 아키텍처가 feature를 공유함")
    print("    → '출력은 직교인데 feature는 공유' 가설 성립 → 공유 부분공간 공격 근거 확보")
    print("  - CKA가 전부 낮으면(≈0) → feature도 공유 적음 → 방향 재검토 필요")

    # 히트맵 저장
    OUT_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(5, 4))
    im = plt.imshow(M, cmap="viridis", vmin=0, vmax=max(0.3, M.max()))
    plt.colorbar(im, label="linear CKA")
    plt.xticks(range(args.layers), [f"D{j}" for j in range(args.layers)])
    plt.yticks(range(args.layers), [f"Y{i}" for i in range(args.layers)])
    plt.xlabel("RT-DETR layer"); plt.ylabel("YOLOv8 layer")
    plt.title("Feature similarity (CKA): YOLOv8 vs RT-DETR")
    for i in range(args.layers):
        for j in range(args.layers):
            plt.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                     color="white" if M[i, j] < 0.25 else "black", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cka_heatmap.png", dpi=150)
    print("\n저장: output/cka_heatmap.png")

    # json 저장
    with open(OUT_DIR / "cka_metrics.json", "w") as f:
        json.dump({"cka_matrix": M.tolist(),
                   "diag_mean": float(np.mean(diag)),
                   "max": float(M.max()), "mean": float(M.mean()),
                   "n_images": args.n, "layers": args.layers}, f, indent=2)
    print("저장: output/cka_metrics.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200, help="분석 이미지 수")
    p.add_argument("--layers", type=int, default=4, help="모델당 hook 층 수")
    p.add_argument("--batch", type=int, default=16, help="배치 크기")
    args = p.parse_args()
    main(args)
