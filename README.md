# FJSP Transformer 專案

基於 Transformer 架構的彈性工作車間排程問題 (Flexible Job Shop Scheduling Problem, FJSP) 解決方案。

## 🎯 專案概念

### 什麼是這個專案？

**簡單說**：我們教會了 AI 如何當一個超級工廠經理

想像一個工廠場景：
- 🏭 有 5 台不同的機器
- 📋 有 10 個客戶訂單要完成  
- ⚙️ 每個訂單需要多個加工步驟
- 🤔 問題：如何安排才能最快完成所有訂單？

### Transformer 的作用

```
原本的 Transformer：
中文句子 → Transformer → 英文句子

我們的 Transformer：
工廠狀態 → Transformer → 最佳決策
```

**核心創新**：把工廠排程問題轉換成語言翻譯問題
- 工廠狀態 = 輸入句子
- 排程決策 = 翻譯結果
- 用 AI 的語言理解能力來解決工業問題

## 🚀 快速開始

### 1. 安裝依賴
```bash
pip install -r requirements.txt
```

### 2. 基礎測試
```bash
python transformer_example.py
```

### 3. FJSP Transformer 訓練
```bash
python train_transformer_final.py
```

## 📁 專案結構

```
fjsp_env/
├── 核心檔案
│   ├── transformer.py              # 基礎 Transformer 實現
│   ├── transformer_example.py      # 基礎 Transformer 測試範例
│   ├── train_transformer_final.py  # FJSP Transformer 訓練主程式
│   ├── fjsp_env.py                # FJSP 環境實現
│   ├── test_gpu.py                # GPU 測試腳本
│   ├── train_gpu.py               # 原始 PPO 訓練腳本
│   ├── validate.py                # 驗證功能
│   └── PPO_model.py               # PPO 模型實現
│
├── 環境模組
│   ├── env/
│   │   ├── __init__.py
│   │   ├── case_generator.py      # FJSP 案例生成器
│   │   └── load_data.py          # 數據載入工具
│   └── utils/                     # 工具目錄
│
├── 配置檔案
│   ├── config.json               # 訓練配置參數
│   └── requirements.txt          # Python 依賴套件
│
├── 數據目錄
│   ├── data_test/                # 測試數據
│   ├── data_dev/                 # 驗證數據
│   └── save/                     # 模型保存目錄
│
└── 說明文件
    └── README.md                 # 本文件
```

## 💻 GPU 使用指南

### 檢查 GPU 可用性
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
```

### 方法 1: 修改現有程式使用 GPU

編輯 `train_transformer_final.py`，找到第 16-17 行：

```python
# 原始代碼 (CPU 版本)
torch.set_default_tensor_type('torch.FloatTensor')
device = torch.device("cpu")
```

改為：

```python
# GPU 版本
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
print(f"使用設備: {device}")
```

### 方法 2: 創建 GPU 專用版本

創建 `train_transformer_gpu.py`：

```python
#!/usr/bin/env python3
"""
GPU 版本的 FJSP Transformer 訓練腳本
"""
import copy
import json
import os
import random
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# GPU 設置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(device)
else:
    torch.set_default_tensor_type('torch.FloatTensor')

print(f"使用設備: {device}")
print(f"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "使用 CPU")

# 其餘代碼與 train_transformer_final.py 相同，但所有張量創建時加上 device 參數
```

### 方法 3: 環境變數控制

```bash
# 強制使用 GPU
export CUDA_VISIBLE_DEVICES=0
python train_transformer_final.py

# 強制使用 CPU  
export CUDA_VISIBLE_DEVICES=""
python train_transformer_final.py

# Windows 用戶
set CUDA_VISIBLE_DEVICES=0
python train_transformer_final.py
```

### GPU 優化建議

1. **增加批次大小**：
```python
# 在 train_transformer_final.py 中修改
batch_size = 16  # 原本是 4，GPU 可以處理更大批次
```

2. **使用混合精度訓練**：
```python
# 加入自動混合精度
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# 在訓練循環中
with autocast():
    loss = agent.train_step(experiences)
```

3. **監控 GPU 使用率**：
```bash
# 另開終端監控
nvidia-smi -l 1
```

### GPU 故障排除

1. **CUDA 記憶體不足**：
```python
# 減少批次大小
batch_size = 2

# 或清理 GPU 記憶體
torch.cuda.empty_cache()
```

2. **設備不匹配錯誤**：
```python
# 確保所有張量都在同一設備
def ensure_device(tensor, device):
    return tensor.to(device) if tensor.device != device else tensor
```

3. **CUDA 版本不匹配**：
```bash
# 重新安裝對應版本的 PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📊 輸入輸出說明

### 輸入
**重要**：程式是自包含的，**不需要準備任何輸入檔案**！

程式內部自動生成：
- 模擬的 FJSP 狀態 (4個批次，5個工作，3台機器)
- 隨機特徵向量 (64維狀態特徵)
- 訓練參數 (25次迭代)

### 輸出檔案
- `save/final_model.pt` - 最終訓練模型
- `save/final_transformer_iter_X.pt` - 各迭代保存的模型
- `save/final_training_results.csv` - 訓練損失記錄

### 預期結果
```
🚀 最終版 FJSP Transformer 訓練
使用設備: cuda:0  # 或 cpu
模型參數數量: 77,994
✅ Iteration 3: Loss = 2.1888
📊 驗證 Makespan: 0.6170
🎊 FJSP Transformer 訓練完全成功！
```

## 🔧 環境需求

### 系統需求
- Python 3.8+
- Windows/Linux/macOS
- 4GB+ RAM (推薦 8GB+)
- NVIDIA GPU (可選，推薦)

### Python 套件
```
torch>=1.12.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
gym>=0.21.0
visdom>=0.1.8
openpyxl>=3.0.9
pynvml>=11.4.1
```

### GPU 需求 (可選)
- NVIDIA GPU (GTX 1060 或更高)
- CUDA 11.8 或更高
- 4GB+ GPU 記憶體

## 🛠️ 故障排除

### 常見問題

1. **CUDA 錯誤**
   ```bash
   # 強制使用 CPU
   export CUDA_VISIBLE_DEVICES=""
   python train_transformer_final.py
   ```

2. **記憶體不足**
   - 減少 `batch_size` (從 4 改為 2)
   - 使用較小的模型參數

3. **套件缺失**
   ```bash
   pip install -r requirements.txt
   ```

### 效能優化

1. **GPU 加速**: 確保 CUDA 正確安裝
2. **批次大小**: 根據記憶體調整 batch_size  
3. **模型大小**: 調整 d_model, n_layers 等參數

## 🎯 技術特點

### Transformer 優勢
- **並行處理**: 比 RNN 更高效的訓練
- **長距離依賴**: 更好地處理序列間關係
- **可解釋性**: 注意力權重提供決策洞察

### FJSP 建模
- **狀態表示**: 將工作車間狀態編碼為序列
- **動作空間**: 操作-機器分配的離散決策
- **獎勵設計**: 基於 makespan 的性能評估

## 📈 應用價值

### 實際應用
- 🏭 **製造業**: 工廠生產排程優化
- 🏥 **醫療**: 手術室和醫生安排
- 🚚 **物流**: 貨車和路線規劃
- ✈️ **航空**: 跑道和航班調度

### 技術創新
- 將 NLP 技術應用到組合優化
- 提供端到端的 AI 排程解決方案
- 展示 Transformer 在非語言領域的潛力

---

**開發完成！** 這個專案提供了一個完整的基於 Transformer 的 FJSP 解決方案，支援 CPU 和 GPU 訓練。