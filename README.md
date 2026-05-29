# 学霸帝 AI 🎓

本地运行的 AI 导师桌面应用，基于 **Qwen3.5-2B** + **Ollama**，无需联网，保护隐私。

## ✨ 功能特性

- 💬 **本地 AI 对话** — 基于 Ollama，完全离线运行
- 📚 **RAG 知识库** — 上传 PDF/Markdown/文本文档，AI 自动参考回答
- 📖 **引导式学习路径** — 粘贴学习材料，AI 自动生成知识点大纲和分步讲解
- 🤔 **思考过程可见** — 支持 Qwen `` 标签，展示推理过程
- 🖥️ **原生 macOS 应用** — PyQt5 构建，DMG 一键安装

## 🚀 快速开始

### 1. 安装 Ollama
```bash
brew install ollama
brew services start ollama
```

### 2. 下载模型
```bash
# 主模型（对话）
ollama create xueba-di -f Modelfile
# 或从 Ollama 库拉取
ollama pull qwen3:2b

# Embedding 模型（知识库需要）
ollama pull nomic-embed-text
```

### 3. 下载应用
前往 [Releases](https://github.com/xuebadi/Tutor/releases) 下载最新 DMG，拖入 Applications 即可。

### 4. 启动
打开「学霸帝 AI」，输入问题开始对话！

## 📚 使用知识库

1. 点击工具栏「📚 知识库」显示侧边栏
2. 点击「📄 上传文档」选择 PDF/MD/TXT 文件
3. 等待索引完成（首次约 10-30 秒）
4. 正常提问，AI 会自动引用知识库内容回答

## 📖 使用引导式学习

1. 点击工具栏「📖 学习路径」
2. 粘贴学习材料全文
3. 点击「✨ 生成学习大纲」
4. 点击大纲节点，逐步学习！

## 🛠️ 从源码构建

```bash
git clone https://github.com/xuebadi/Tutor.git
cd Tutor
pip3 install PyQt5 faiss-cpu PyMuPDF sentence-transformers
python3 main.py
```

打包 DMG：
```bash
pip3 install pyinstaller
pyinstaller 学霸帝AI_v110.spec
hdiutil create -volname "学霸帝AI" -srcfolder dist/学霸帝AI.app -ov -format UDZO dist/学霸帝AI.dmg
```

## 📋 系统要求

- macOS 10.15+
- Ollama 0.20+
- 建议 8GB+ 内存

## 📄 许可证

MIT License
