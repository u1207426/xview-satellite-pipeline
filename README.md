# xView 衛星影像 AI Pipeline

基於 [xView 資料集](http://xviewdataset.org) 的兩階段衛星影像軍事設施偵測與分類系統。

## 架構

```
衛星影像
    │
    ▼
┌─────────────┐
│  L1 偵測器  │  YOLOv11m — 偵測航空器、儲油槽、車輛停放區、建築物
└─────────────┘
    │  偵測結果 + 31 維結構特徵向量
    ▼
┌─────────────┐
│  L2 分類器  │  SatMAE ViT-Base + MLP 融合 — P(軍事設施)
└─────────────┘
    │
    ▼
annotated.jpg · result.json · report.html
```

**L1** 以 640×640 tiles 對 xView 4 個物件類別進行 YOLOv11m fine-tune。  
**L2** 將 ViT 影像特徵與 31 維結構特徵向量（類別計數、空間分布、信心統計）融合，判斷該地點是否為軍事設施。

## 專案結構

```
defence-satellite-pipeline/
├── configs/
│   ├── l1_yolo11.yaml          # L1 訓練超參數
│   ├── l2_classifier.yaml      # L2 訓練超參數
│   └── data_sources.yaml       # 資料集路徑與類別對應
├── src/
│   ├── data/
│   │   ├── tiling.py           # 大圖切割為 640/1024px tiles
│   │   ├── converters.py       # xView GeoJSON → YOLO 標籤格式
│   │   └── dataset.py          # L2 PyTorch Dataset（影像 + 結構特徵）
│   ├── models/
│   │   ├── l1_detector.py      # YOLOv11m wrapper
│   │   ├── l2_classifier.py    # ViT-Base + MLP 融合分類器
│   │   └── feature_extractor.py# 31 維結構特徵提取
│   ├── training/
│   │   ├── train_l1.py         # L1 訓練入口
│   │   └── train_l2.py         # L2 訓練 + L1 特徵快取
│   ├── evaluation/
│   │   └── metrics.py          # AUC-ROC、F1、混淆矩陣
│   └── inference/
│       └── predict.py          # 單張影像端到端推論
├── notebooks/
│   └── colab_training.ipynb    # Google Colab T4 完整訓練流程
└── tests/                      # 42 個 pytest 測試
```

## 快速開始（本地）

### 環境需求

```bash
pip install -r requirements.txt
```

需要 Python 3.9+、PyTorch 2.0+，建議使用 CUDA GPU。

### 1. 準備資料

前往 [xviewdataset.org](http://xviewdataset.org) 申請下載 xView 資料集（需填表申請），解壓後放置於：

```
data/raw/xview/
├── train_images/
└── xView_train.geojson
```

轉換標籤格式並切割影像：

```bash
python - <<'EOF'
from src.data.converters import convert_xview_to_yolo
from src.data.tiling import tile_image

convert_xview_to_yolo(
    geojson_path="data/raw/xview/xView_train.geojson",
    image_dir="data/raw/xview/train_images",
    output_dir="data/annotations/yolo_format",
)
EOF
```

### 2. L1 訓練（YOLOv11m）

```bash
python -m src.training.train_l1 \
  --config configs/l1_yolo11.yaml \
  --data data/xview_dataset.yaml
```

最佳 checkpoint 儲存於 `checkpoints/l1/weights/best.pt`。

### 3. 快取 L1 特徵並訓練 L2

```bash
python -m src.training.train_l2 \
  --config configs/l2_classifier.yaml \
  --train-images data/processed/tiles_1024/train \
  --val-images   data/processed/tiles_1024/val \
  --features-dir data/annotations/l2_features \
  --train-labels data/annotations/location_labels/train.csv \
  --val-labels   data/annotations/location_labels/val.csv
```

L2 標籤 CSV 格式：

```
image_name,label
1102.tif,0
1234.tif,1
```

`1` = 軍事設施，`0` = 非軍事。

### 4. 推論

```bash
python -m src.inference.predict \
  --image path/to/image.tif \
  --l1-checkpoint checkpoints/l1/weights/best.pt \
  --l2-checkpoint checkpoints/l2/best.pt
```

結果輸出至 `outputs/<location_id>_<timestamp>/`：

| 檔案 | 內容 |
|---|---|
| `annotated.jpg` | 標有 L1 偵測框的影像 |
| `result.json` | 完整偵測結果 + L2 機率 |
| `report.html` | 人工可讀分析報告 |

**信心閾值規則：**

| P(軍事) | 分類 | 等級 |
|---|---|---|
| ≥ 0.8 | 軍事設施 | 高 |
| 0.6 – 0.8 | 軍事設施 | 中 |
| 0.4 – 0.6 | 不確定 | 低 |
| < 0.4 | 非軍事 | — |

## Google Colab 訓練

以 T4 GPU runtime 開啟 `notebooks/colab_training.ipynb`。Notebook 已內建 Drive 掛載、套件安裝、影像切割、L1/L2 訓練及斷線 resume 等完整流程。

T4 GPU 預估時間：

| 步驟 | 預估時間 |
|---|---|
| 影像切割（CPU） | 30–60 分鐘 |
| L1 訓練（100 epochs） | 2–4 小時 |
| L1 特徵快取 | 20–40 分鐘 |
| L2 訓練（50 epochs） | 30–60 分鐘 |

## 模型細節

### L1 — YOLOv11m

| 參數 | 設定值 |
|---|---|
| 基礎模型 | `yolo11m.pt`（COCO 預訓練） |
| 偵測類別 | 航空器、儲油槽、車輛停放區、建築物 |
| 輸入尺寸 | 640 × 640 |
| 凍結層數 | 前 10 層（暖機階段） |
| 資料增強 | mosaic、上下翻轉、45° 旋轉、HSV 色彩抖動 |

### L2 — SatMAE ViT-Base + MLP 融合

| 參數 | 設定值 |
|---|---|
| 影像骨幹 | `facebook/vit-mae-base`（可替換為 SatMAE 權重） |
| 結構特徵輸入 | 31 維向量（類別計數、空間統計、信心統計） |
| 凍結策略 | 前 5 epoch 凍結骨幹，之後解凍最後 4 個 block |
| 損失函數 | BCEWithLogitsLoss，pos_weight = 3.0 |
| 骨幹學習率 | 1e-5 |
| 分類頭學習率 | 1e-3 |

31 維結構特徵向量組成：
- 13 維：各類別偵測數量與平均信心值
- 4 維：質心位置與空間分布統計
- 13 維：信心值區間計數
- 1 維：總偵測數量

## 測試

```bash
pytest
```

42 個測試，約 80 秒。

## 資料集

本專案使用 [xView](http://xviewdataset.org)（DIUx，2018），為包含 100 萬個以上標註實例、涵蓋 60 個類別的衛星影像物件偵測資料集。使用前需完成註冊並同意 xView 使用條款。

## 授權

僅供學術研究使用。xView 資料集使用須遵守 [xView 使用條款](http://xviewdataset.org)。
