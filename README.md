# Tutor 🎓

**学霸帝 AI** — 本地运行的 AI 导师桌面应用，保护隐私，无需联网。

基于 **PyQt5 + Ollama + Qwen3.5-2B** 构建，支持 macOS（Intel / Apple Silicon）。

---

## ✨ 功能特性

- 🔒 **100% 本地运行** — 数据不出本机，保护隐私
- 🧠 **Qwen3.5-2B** — 轻量级高性能本地 LLM
- 💬 **流式响应** — 实时显示 AI 思考过程
- 🎨 **深色主题 UI** — 护眼设计，消息气泡式对话
- 🔄 **思考模式** — 展示 AI 推理过程（`<think>` 标签）
- 📦 **DMG 一键安装** — 打包为 macOS 原生应用

---

## 📦 安装

### 1. 安装 Ollama

```bash
brew install ollama
brew services start ollama
```

### 2. 下载模型

```bash
# 方式一：从 Ollama 库拉取（推荐）
ollama pull qwen3.5:2b

# 方式二：导入自定义 GGUF
ollama create xueba-di -f Modelfile
```

### 3. 下载应用

从 [Releases](https://github.com/xuebadi/Tutor/releases) 页面下载 `学霸帝AI.dmg`，双击安装。

---

## 🛠 从源码构建

### 依赖

```bash
pip3 install PyQt5 pyinstaller
brew install ollama
```

### 运行

```bash
python3 main.py
```

### 打包 DMG

```bash
# 使用 PyInstaller
pyinstaller pyinstaller.spec

# 创建 DMG
hdiutil create -volname "学霸帝AI" \
  -srcfolder dist/学霸帝AI.app \
  -ov -format UDZO dist/学霸帝AI.dmg
```

---

## 🚀 使用

1. 启动 Ollama 服务：`brew services start ollama`
2. 打开「学霸帝 AI」应用
3. 底部输入框输入问题，按 Enter 发送
4. 查看 AI 实时思考过程和最终回答

---

## 📁 项目结构

```
Tutor/
├── main.py              # 主程序（PyQt5 GUI + Ollama API）
├── pyinstaller.spec     # PyInstaller 打包配置
├── assets/              # 图标等资源
└── xueba_di_ai_plan.md # 技术方案文档
```

---

## ⚠️ 已知问题

- **Intel Mac 推理速度慢** — CPU 推理约 20-40 秒/回答，建议 Apple Silicon Mac
- **模型需手动导入** — 首次使用需自行下载 Qwen3.5-2B 模型

---

## 📄 许可

MIT License

---

**作者**: [@xuebadi](https://github.com/xuebadi)
