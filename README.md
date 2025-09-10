# Vision_World 🪐

**Vision_World** is a self-built computer vision toolkit that unifies  
state-of-the-art training tricks into a clean, extensible engine.  
It’s designed for **fast prototyping, efficient large-scale training, and easy deployment**.  

---

## ✨ Features
- 🔧 **Modular design**: `Model/`, `Train/`, `Utils/`
- ⚡ **Distributed training** with PyTorch DDP or DeepSpeed
- 🧮 **Mixed precision** (fp16 / bf16) with `torch.amp`
- 🔄 **Exponential Moving Average (EMA)** & **Stochastic Weight Averaging (SWA)**
- ➕ **Gradient Accumulation** for large effective batch sizes on limited GPUs
- 🔒 **CUDA Graph capture** for faster, more deterministic training steps
- 🛠️ **Model registry** (ResNet, PyramidNet, ConvNeXt, Vision Transformer, …)
- 📊 Built-in metrics (Top-1/Top-5 accuracy, RMSE, mAP, …)
- 📦 **ONNX export** + TensorRT deployment ready
- 📝 **TQDM-safe logging** (console + file, TensorBoard/W&B optional)
- ✅ **Hooks** for gradient checkpointing, activation compression, or custom quantization

---

## 🚀 Quickstart

### 1. Clone & install
```bash
git clone https://github.com/YitianYu69/Vision_World.git
cd Vision_World
pip install -e .


