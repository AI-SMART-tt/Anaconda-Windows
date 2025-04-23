太好了！你使用的是 **Windows 版本**，以下是针对 **Windows 系统下的 Anaconda 安装、环境创建和包安装**的**完整、验证后的详细流程**，让你可以顺利创建并使用名为 `stt_torch` 的 Python 环境。

---

# 🐍 一、在 Windows 上安装 Anaconda

### ✅ 1. 下载 Anaconda

- 打开官网：[https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
- 点击 “Download”，选择：
  - 操作系统：**Windows**
  - 版本：**64-bit Graphical Installer**
  - 语言：**Python 3.10 或 3.11**（建议使用 3.10，兼容性更好）

### ✅ 2. 安装 Anaconda

- 双击下载的 `.exe` 文件；
- 点击 “Next”；
- 建议选择 “**Just Me**” 安装；
- 默认安装路径建议保留；
- **⚠️ 勾选：Add Anaconda to my PATH environment variable**（虽然有提示不推荐，但为了方便命令行操作，可以勾选）；
- 安装完成后，点击 “Finish”。

---

# ✅ 二、打开 Anaconda Prompt 并创建环境

### ✅ 1. 打开 Anaconda Prompt

- 点击开始菜单 -> 搜索 “Anaconda Prompt” -> 右键以管理员身份运行（推荐）

### ✅ 2. 创建名为 `stt_torch` 的新环境

```bash
conda create -n stt_torch python=3.10 -y
```

- `-n stt_torch`：环境名称；
- `python=3.10`：指定 Python 版本；
- `-y`：自动确认安装。

### ✅ 3. 激活该环境

```bash
conda activate stt_torch
```

激活后，命令行前缀变为 `(stt_torch)`，说明环境已切换成功。

---

# ✅ 三、在 stt_torch 环境中安装所需包

## 📦 安装 PyTorch（官方推荐方式）

### ✅ 1. 打开 PyTorch 官网安装命令生成器：

👉 https://pytorch.org/get-started/locally/

选择：
- OS: Windows
- Package: Conda
- Language: Python
- Compute Platform: CUDA（GPU 用户）或 CPU

### ✅ 2. 示例安装命令：

#### 👉 如果你使用 **CPU**：

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

#### 👉 如果你使用 **NVIDIA GPU + CUDA 11.8**：

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

> ⚠️ 注意：使用 GPU 需要你的电脑已安装对应版本的 NVIDIA 驱动。

---

## 📦 安装语音识别相关包

在 `stt_torch` 环境中运行：

```bash
pip install librosa soundfile transformers jiwer
```

这些包用途：
- `librosa`：音频处理
- `soundfile`：音频读写
- `transformers`：HuggingFace 预训练模型（如 Whisper）
- `jiwer`：评估语音识别准确率（如计算 WER）

---

## 📦 安装数据分析与开发工具：

```bash
conda install numpy scipy pandas matplotlib scikit-learn jupyter -y
```

---

# ✅ 四、验证安装成功

### ✅ 1. 进入 Python 解释器：

```bash
python
```

然后输入以下代码：

```python
import torch
print("PyTorch 版本:", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())
```

**输出示例（CPU）：**

```
PyTorch 版本: 2.2.0
CUDA 是否可用: False
```

**输出示例（GPU）：**

```
PyTorch 版本: 2.2.0
CUDA 是否可用: True
```

输入 `exit()` 回车退出。

---

# ✅ 五、在 Jupyter Notebook 中使用该环境（可选）

若你使用 Jupyter Notebook，可以注册该环境为内核：

```bash
python -m ipykernel install --user --name=stt_torch --display-name "Python (stt_torch)"
```

完成后，在 Jupyter 中就能选择 “Python (stt_torch)” 作为内核运行代码了。

---

# ✅ 六、环境管理（常用命令）

| 功能 | 命令 |
|------|------|
| 查看所有环境 | `conda env list` |
| 删除环境 | `conda remove -n stt_torch --all` |
| 导出当前环境 | `conda env export > stt_torch_env.yml` |
| 从文件创建环境 | `conda env create -f stt_torch_env.yml` |

---

