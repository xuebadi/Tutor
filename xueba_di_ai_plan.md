# 学霸帝 AI - 本地导师应用技术方案

## 项目状态：进行中 ⚠️

## 一、已完成的工作

### 环境准备 ✅
- [x] Python 3.11.15 安装 (Homebrew)
- [x] llama-cpp-python 安装 (v0.3.23)
- [x] Ollama 安装 (v0.24.0)

### 模型下载 ⚠️
- [x] Qwen3.5-2B-MTP-Q4_K_M.gguf (1.2GB) - 已下载，但与 llama.cpp 不兼容
- [x] mmproj-BF16.gguf (640MB) - 已下载
- [ ] Qwen2.5-2B-Instruct - 因网络问题未能下载

## 二、遇到的问题

### 模型兼容性问题
**问题**: Qwen3.5-2B-MTP 使用了新的 SSM (State Space Model) 架构，当前 llama-cpp-python 版本不完全支持。

**错误信息**:
```
llama_model_load: error loading model: missing tensor 'blk.24.ssm_conv1d.weight'
```

**解决方案选择**:
1. 等待 llama.cpp 更新支持 Qwen3.5 MTP
2. 使用 Qwen2.5 标准模型替代（推荐）
3. 使用 Ollama 运行时

## 三、技术架构

```
┌─────────────────────────────────────────────────────────┐
│                  macOS App (DMG)                       │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │  Frontend    │  │   Backend    │  │  Ollama    │ │
│  │  (Next.js)   │◄─┤  (Python)   │◄─┤  Runtime   │ │
│  │  Port 3782   │  │  Port 8001   │  │  (Local)   │ │
│  └──────────────┘  └──────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## 四、实施计划

### 方案 A: 使用标准 Qwen2.5 模型 (推荐)

需要用户手动操作：
```bash
# 手动下载 Qwen2.5-2B 模型 (约1.5GB)
# 方案1: 从 Ollama 库拉取
ollama pull qwen2.5:2b

# 方案2: 使用 LM Studio (推荐 GUI)
# https://lmstudio.ai/
```

### 方案 B: 使用 Ollama API

DeepTutor 可以通过 Ollama API 连接本地模型：
- Ollama 默认端口: 11434
- DeepTutor 配置本地 LLM provider

## 五、下一步操作

1. 用户需要手动安装 LM Studio 或使用 Ollama 拉取模型
2. 配置 DeepTutor 连接本地模型
3. 构建 macOS DMG 应用

## 六、已安装的工具路径

- Python 3.11: `/usr/local/opt/python@3.11/bin/python3.11`
- Ollama: `/usr/local/opt/ollama/bin/ollama`
- 模型目录: `~/.qclaw/workspace-ua58rsb93veqtxl7/models/`

---

## 快速开始指南 (用户)

### 方式 1: 使用 LM Studio (推荐)
1. 下载 LM Studio: https://lmstudio.ai/
2. 搜索并下载 "Qwen2.5-2B"
3. 运行模型并确保 API 在 http://localhost:1234/v1

### 方式 2: 使用 Ollama
```bash
# 启动 Ollama
brew services start ollama

# 拉取模型
ollama pull qwen2.5:2b

# 测试
ollama run qwen2.5:2b "你好"
```

完成上述步骤后，告诉我继续配置 DeepTutor。
