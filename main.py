"""
学霸帝 AI v1.1.0 - macOS 本地导师应用
基于 PyQt5 + Ollama Qwen3.5-2B 本地模型
新增：RAG 知识库 + 引导式学习路径
"""
import os
os.environ["PYQT5_NO_ABORT"] = "1"

import sys
import json
import threading
import urllib.request
import urllib.error
import rag_engine
import learning_path
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QLineEdit, QScrollArea,
    QFrame, QSplitter, QStatusBar, QComboBox, QAction, QToolBar,
    QMessageBox, QGraphicsDropShadowEffect, QListWidget, QListWidgetItem,
    QFileDialog, QProgressBar, QTabWidget, QDialog, QFormLayout,
    QSpinBox, QTextBrowser, QTreeWidget, QTreeWidgetItem, QDialogButtonBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer, QUrl
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon, QTextCursor, QTextCharFormat, QCursor


# ============== 配置 ==============
OLLAMA_API    = "http://localhost:11434"
DEFAULT_MODEL = "xueba-di"
APP_NAME      = "学霸帝 AI"
APP_VERSION   = "1.1.0"

# ============== 颜色主题 ==============
COLOR_BG            = "#1a1a2e"
COLOR_SURFACE      = "#16213e"
COLOR_USER_BG      = "#0f3460"
COLOR_AI_BG        = "#1e3a5f"
COLOR_ACCENT       = "#e94560"
COLOR_TEXT          = "#eaeaea"
COLOR_TEXT_SECONDARY = "#8892b0"
COLOR_BORDER        = "#2a3f5f"
COLOR_INPUT_BG     = "#0f3460"
COLOR_SUCCESS       = "#4ade80"
COLOR_WARNING       = "#fbbf24"


# ============== 流式响应线程 ==============
class OllamaStreamThread(QThread):
    response_ready  = pyqtSignal(str)
    thinking_ready = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    token_count     = pyqtSignal(int)

    def __init__(self, prompt, model=DEFAULT_MODEL, parent=None):
        super().__init__(parent)
        self.prompt = prompt
        self.model  = model
        self._running = True

    def run(self):
        try:
            # 检查模型
            try:
                req = urllib.request.Request(
                    f"{OLLAMA_API}/api/show",
                    data=json.dumps({"name": self.model}).encode(),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    pass
            except Exception:
                self.finished_signal.emit(False, f"❌ 模型 '{self.model}' 未找到。请先运行：ollama create xueba-di")
                return

            req = urllib.request.Request(
                f"{OLLAMA_API}/api/chat",
                data=json.dumps({
                    "model": self.model,
                    "messages": [{"role": "user", "content": self.prompt}],
                    "stream": True,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 1024,
                        "num_ctx": 2048,
                    }
                }).encode(),
                headers={"Content-Type": "application/json"}
            )

            token_count  = 0
            response_text = ""
            in_think      = False
            pending       = ""
            think_buf     = ""
            answer_buf    = ""

            with urllib.request.urlopen(req, timeout=300) as resp:
                for raw_line in resp:
                    if not self._running:
                        break
                    try:
                        data = json.loads(raw_line)
                        content = data.get("message", {}).get("content", "")
                        if not content:
                            continue

                        pending += content
                        while pending:
                            if not in_think:
                                idx = pending.find("<think>")
                                if idx >= 0:
                                    # 标签前的文本 → 回答
                                    before = pending[:idx]
                                    answer_buf += before
                                    if answer_buf.strip():
                                        response_text += answer_buf
                                        self.response_ready.emit(answer_buf)
                                        token_count += 1
                                        self.token_count.emit(token_count)
                                        answer_buf = ""
                                    pending = pending[idx + 7:]
                                    in_think = True
                                else:
                                    # 没有完整标签，保留最后 6 字符防止标签跨 chunk
                                    if len(pending) > 6:
                                        chunk = pending[:-6]
                                        pending = pending[-6:]
                                        answer_buf += chunk
                                    break
                            else:
                                idx = pending.find("</think>")
                                if idx >= 0:
                                    think_buf += pending[:idx]
                                    if think_buf.strip():
                                        self.thinking_ready.emit(think_buf)
                                    pending = pending[idx + 8:]
                                    in_think = False
                                    think_buf = ""
                                else:
                                    if len(pending) > 8:
                                        chunk = pending[:-8]
                                        pending = pending[-8:]
                                        think_buf += chunk
                                    break

                        # flush
                        if answer_buf and not in_think:
                            response_text += answer_buf
                            self.response_ready.emit(answer_buf)
                            token_count += 1
                            self.token_count.emit(token_count)
                            answer_buf = ""

                    except json.JSONDecodeError:
                        continue

            if response_text.strip():
                self.finished_signal.emit(True, "")
            else:
                self.finished_signal.emit(False, "模型未返回回答内容，请重试")

        except urllib.error.URLError:
            self.finished_signal.emit(False, "❌ 连接失败：Ollama 服务未启动\n\n请在终端运行：brew services start ollama")
        except Exception as e:
            self.finished_signal.emit(False, f"❌ 错误：{str(e)}")

    def stop(self):
        self._running = False


# ============== RAG 建索引线程 ==============
class RAGIndexThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, file_paths):
        super().__init__()
        self.file_paths = file_paths

    def run(self):
        try:
            total = 0
            for path in self.file_paths:
                self.progress.emit(f"正在索引：{os.path.basename(path)}")
                n = rag_engine.add_document(path)
                total += n
                self.progress.emit(f"✅ {os.path.basename(path)} → {n} 个片段")
            self.finished.emit(True, f"知识库更新完成，新增 {total} 个文本片段")
        except Exception as e:
            self.finished.emit(False, f"索引失败：{str(e)}")


# ============== 消息气泡 ==============
class MessageBubble(QFrame):
    def __init__(self, text, is_user=False, is_thinking=False, parent=None):
        super().__init__(parent)
        self.is_user     = is_user
        self.is_thinking = is_thinking
        self.setup_ui(text)

    def setup_ui(self, text):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        if self.is_user:
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: {COLOR_USER_BG};
                    border-radius: 16px;
                    border: 1px solid {COLOR_ACCENT};
                }}
            """)
            label = QLabel(f"<span style='color:{COLOR_TEXT}'>{self.escape_html(text)}</span>")
        elif self.is_thinking:
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: #1a1a2e;
                    border-radius: 12px;
                    border: 1px dashed #3a3a5e;
                }}
            """)
            label = QLabel(f"<span style='color:#6a6a9a; font-style: italic;'>🤔 {self.escape_html(text)}</span>")
        else:
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: {COLOR_AI_BG};
                    border-radius: 16px;
                    border: 1px solid {COLOR_BORDER};
                }}
            """)
            label = QLabel(f"<span style='color:{COLOR_TEXT}'>{self.escape_html(text)}</span>")

        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(label)

    def update_text(self, text):
        for child in self.findChildren(QLabel):
            if self.is_thinking:
                child.setText(f"<span style='color:#6a6a9a; font-style: italic;'>🤔 {self.escape_html(text)}</span>")
            else:
                child.setText(f"<span style='color:{COLOR_TEXT}'>{self.escape_html(text)}</span>")

    @staticmethod
    def escape_html(text):
        return (text.replace("&", "&amp;")
                 .replace("<", "&lt;")
                 .replace(">", "&gt;")
                 .replace("\n", "<br>"))


# ============== 知识库侧边栏 ==============
class KnowledgeSidebar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(260)
        self.setStyleSheet(f"background:{COLOR_SURFACE}; border-right:1px solid {COLOR_BORDER};")
        self.setup_ui()
        self.refresh_status()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel(f"<span style='color:{COLOR_ACCENT}; font-size:15px; font-weight:bold;'>📚 知识库</span>")
        layout.addWidget(title)

        # 状态
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color:{COLOR_TEXT_SECONDARY}; font-size:12px;")
        layout.addWidget(self.status_label)

        # 文档列表
        self.doc_list = QListWidget()
        self.doc_list.setStyleSheet(f"""
            QListWidget {{
                background: {COLOR_BG};
                border: 1px solid {COLOR_BORDER};
                border-radius: 8px;
                color: {COLOR_TEXT};
                font-size: 12px;
            }}
            QListWidget::item {{
                padding: 6px;
                border-bottom: 1px solid {COLOR_BORDER};
            }}
            QListWidget::item:selected {{
                background: {COLOR_ACCENT};
            }}
        """)
        layout.addWidget(self.doc_list, 1)

        # 按钮区
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(6)

        self.upload_btn = QPushButton("📄 上传文档")
        self.upload_btn.setStyleSheet(self._btn_style(COLOR_ACCENT))
        self.upload_btn.clicked.connect(self.upload_files)
        btn_layout.addWidget(self.upload_btn)

        self.refresh_btn = QPushButton("🔄 刷新状态")
        self.refresh_btn.setStyleSheet(self._btn_style(COLOR_BORDER))
        self.refresh_btn.clicked.connect(self.refresh_status)
        btn_layout.addWidget(self.refresh_btn)

        self.clear_btn = QPushButton("🗑️ 清空知识库")
        self.clear_btn.setStyleSheet(self._btn_style("#5a2020"))
        self.clear_btn.clicked.connect(self.confirm_clear)
        btn_layout.addWidget(self.clear_btn)

        layout.addLayout(btn_layout)

    def _btn_style(self, bg):
        return f"""
            QPushButton {{
                background: {bg};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px;
                font-size: 12px;
            }}
            QPushButton:hover {{ background: {COLOR_ACCENT}; }}
        """

    def refresh_status(self):
        try:
            status = rag_engine.kb_status()
            docs   = status.get("docs", {})
            chunks = status["total_chunks"]
            ready  = status.get("embed_ready", False)
            embed_color = COLOR_SUCCESS if ready else COLOR_WARNING
            embed_text  = "就绪" if ready else "未就绪（请运行 ollama pull nomic-embed-text）"

            html = f"""
            <span style='color:{COLOR_TEXT_SECONDARY};'>
                文档数：<b style='color:{COLOR_TEXT}'>{len(docs)}</b><br>
                文本片段：<b style='color:{COLOR_TEXT}'>{chunks}</b><br>
                Embedding：<b style='color:{embed_color}'>● {embed_text}</b>
            </span>
            """
            self.status_label.setText(html)

            self.doc_list.clear()
            for doc_name, cnt in docs.items():
                item = QListWidgetItem(f"{doc_name} ({cnt} 片段)")
                item.setData(Qt.UserRole, doc_name)
                self.doc_list.addItem(item)
        except Exception as e:
            self.status_label.setText(f"<span style='color:{COLOR_WARNING}'>⚠️ {str(e)[:50]}</span>")

    def upload_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择文档", os.path.expanduser("~"),
            "文档 (*.pdf *.md *.txt);;PDF (*.pdf);;Markdown (*.md *.markdown);;文本 (*.txt)"
        )
        if not files:
            return

        self.upload_btn.setEnabled(False)
        self.upload_btn.setText("⏳ 索引中...")

        self.index_thread = RAGIndexThread(files)
        self.index_thread.progress.connect(self._on_progress)
        self.index_thread.finished.connect(self._on_index_done)
        self.index_thread.start()

    def _on_progress(self, msg):
        self.status_label.setText(f"<span style='color:{COLOR_TEXT_SECONDARY}'>{msg}</span>")

    def _on_index_done(self, ok, msg):
        self.upload_btn.setEnabled(True)
        self.upload_btn.setText("📄 上传文档")
        self.refresh_status()
        if ok:
            QMessageBox.information(self, "知识库", msg)
        else:
            QMessageBox.warning(self, "错误", msg)

    def confirm_clear(self):
        if QMessageBox.question(self, "确认", "确定要清空整个知识库吗？") == QMessageBox.Yes:
            rag_engine.clear_kb()
            self.refresh_status()


# ============== 引导式学习路径对话框 ==============
class LearningPathDialog(QDialog):
    """弹出式引导学习窗口"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.outline  = []
        self.current_id = 1
        self.material_text = ""
        self.setWindowTitle("📖 引导式学习")
        self.setMinimumSize(700, 550)
        self.setStyleSheet(f"background:{COLOR_BG}; color:{COLOR_TEXT};")
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # 顶部：粘贴学习材料
        top_frame = QFrame()
        top_frame.setStyleSheet(f"background:{COLOR_SURFACE}; border-radius:12px; padding:12px;")
        top_layout = QVBoxLayout(top_frame)

        hint = QLabel(f"<span style='color:{COLOR_TEXT_SECONDARY}'>粘贴学习材料内容，AI 将自动生成知识点大纲和引导式讲解：</span>")
        top_layout.addWidget(hint)

        self.material_edit = QTextEdit()
        self.material_edit.setPlaceholderText("粘贴学习材料全文...")
        self.material_edit.setMaximumHeight(100)
        self.material_edit.setStyleSheet(f"""
            QTextEdit {{
                background: {COLOR_BG};
                color: {COLOR_TEXT};
                border: 1px solid {COLOR_BORDER};
                border-radius: 8px;
                padding: 8px;
                font-size: 13px;
            }}
        """)
        top_layout.addWidget(self.material_edit)

        btn_row = QHBoxLayout()
        self.gen_outline_btn = QPushButton("✨ 生成学习大纲")
        self.gen_outline_btn.setStyleSheet(f"background:{COLOR_ACCENT}; color:white; border:none; border-radius:8px; padding:8px 16px;")
        self.gen_outline_btn.clicked.connect(self.generate_outline)
        btn_row.addWidget(self.gen_outline_btn)
        btn_row.addStretch()
        top_layout.addLayout(btn_row)

        layout.addWidget(top_frame)

        # 中间：大纲树 + 讲解区
        mid_splitter = QSplitter(Qt.Horizontal)

        # 左侧大纲
        self.outline_tree = QTreeWidget()
        self.outline_tree.setHeaderLabel("📋 学习大纲")
        self.outline_tree.setStyleSheet(f"""
            QTreeWidget {{
                background: {COLOR_SURFACE};
                color: {COLOR_TEXT};
                border: 1px solid {COLOR_BORDER};
                border-radius: 8px;
                font-size: 13px;
            }}
        """)
        self.outline_tree.itemClicked.connect(self.on_outline_click)
        self.outline_tree.setMinimumWidth(200)
        mid_splitter.addWidget(self.outline_tree)

        # 右侧讲解
        self.lesson_view = QTextBrowser()
        self.lesson_view.setOpenExternalLinks(True)
        self.lesson_view.setStyleSheet(f"""
            QTextBrowser {{
                background: {COLOR_SURFACE};
                color: {COLOR_TEXT};
                border: 1px solid {COLOR_BORDER};
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
            }}
        """)
        mid_splitter.addWidget(self.lesson_view)
        mid_splitter.setStretchFactor(1, 1)
        layout.addWidget(mid_splitter, 1)

        # 底部：导航
        nav = QHBoxLayout()
        self.prev_btn = QPushButton("◀ 上一步")
        self.prev_btn.setEnabled(False)
        self.prev_btn.setStyleSheet(f"background:{COLOR_BORDER}; color:{COLOR_TEXT}; border:none; border-radius:8px; padding:8px 16px;")
        self.prev_btn.clicked.connect(self.go_prev)

        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet(f"color:{COLOR_TEXT_SECONDARY};")
        self.progress_label.setAlignment(Qt.AlignCenter)

        self.next_btn = QPushButton("下一步 ▶")
        self.next_btn.setEnabled(False)
        self.next_btn.setStyleSheet(f"background:{COLOR_ACCENT}; color:white; border:none; border-radius:8px; padding:8px 16px;")
        self.next_btn.clicked.connect(self.go_next)

        nav.addWidget(self.prev_btn)
        nav.addStretch()
        nav.addWidget(self.progress_label)
        nav.addStretch()
        nav.addWidget(self.next_btn)
        layout.addLayout(nav)

    def generate_outline(self):
        text = self.material_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "提示", "请先粘贴学习材料！")
            return

        self.material_text = text
        self.gen_outline_btn.setEnabled(False)
        self.gen_outline_btn.setText("生成中...")

        # 后台线程调用 Ollama 生成大纲
        def do_work():
            import learning_path
            self.outline = learning_path.extract_outline(text)
            return self.outline

        def on_done(fut):
            try:
                outline = fut.result() if hasattr(fut, 'result') else do_work()
                self.outline = outline if isinstance(outline, list) else []
                self._populate_outline()
            except Exception as e:
                QMessageBox.warning(self, "错误", f"生成大纲失败：{e}")
            finally:
                self.gen_outline_btn.setEnabled(True)
                self.gen_outline_btn.setText("✨ 生成学习大纲")

        # 用 threading 避免阻塞 UI
        def worker():
            try:
                outline = do_work()
                QTimer.singleShot(0, lambda: self._on_outline_ready(outline))
            except Exception as e:
                QTimer.singleShot(0, lambda: QMessageBox.warning(self, "错误", f"生成大纲失败：{e}"))
                QTimer.singleShot(0, lambda: (
                    self.gen_outline_btn.setEnabled(True),
                    self.gen_outline_btn.setText("✨ 生成学习大纲")
                ))

        threading.Thread(target=worker, daemon=True).start()

    def _on_outline_ready(self, outline):
        self.outline = outline if isinstance(outline, list) else []
        self._populate_outline()
        self.gen_outline_btn.setEnabled(True)
        self.gen_outline_btn.setText("✨ 重新生成大纲")

    def _populate_outline(self):
        self.outline_tree.clear()
        for item in self.outline:
            tree_item = QTreeWidgetItem(self.outline_tree)
            tree_item.setText(0, f"{item['id']}. {item['title']}")
            tree_item.setData(0, Qt.UserRole, item["id"])
            if item.get("summary"):
                child = QTreeWidgetItem(tree_item)
                child.setText(0, f"  {item['summary']}")
        if self.outline:
            self.current_id = 1
            self._load_lesson()
            self.next_btn.setEnabled(True)

    def on_outline_click(self, item, col):
        data = item.data(0, Qt.UserRole)
        if data:
            self.current_id = data
            self._load_lesson()

    def _load_lesson(self):
        if not self.outline:
            return
        item = next((o for o in self.outline if o["id"] == self.current_id), None)
        if not item:
            return

        self.lesson_view.setHtml(f"""
        <div style='color:{COLOR_TEXT_SECONDARY};'>
            ⏳ 正在生成「{item['title']}」的讲解...
        </div>
        """)

        self.prev_btn.setEnabled(self.current_id > 1)
        self.next_btn.setEnabled(self.current_id < len(self.outline))
        self.progress_label.setText(f"第 {self.current_id} / {len(self.outline)} 步")

        # 后台生成讲解
        def worker():
            import learning_path
            result = learning_path.generate_lesson(self.outline, self.current_id, self.material_text)
            QTimer.singleShot(0, lambda: self._on_lesson_ready(result))

        threading.Thread(target=worker, daemon=True).start()

    def _on_lesson_ready(self, result):
        if "error" in result:
            self.lesson_view.setHtml(f"<div style='color:{COLOR_ACCENT}'>{result['error']}</div>")
            return
        html = f"""
        <h2 style='color:{COLOR_ACCENT}'>{result['title']}</h2>
        <div style='line-height:1.8;'>{result['content'].replace(chr(10), '<br>')}</div>
        <hr style='border-color:{COLOR_BORDER}'>
        <div style='color:{COLOR_TEXT_SECONDARY}; font-size:12px;'>
            进度：{result['current']} / {result['total']} |
            <a href='prev' style='color:{COLOR_ACCENT}'>上一步</a> |
            <a href='next' style='color:{COLOR_ACCENT}'>下一步</a>
        </div>
        """
        self.lesson_view.setHtml(html)

    def go_prev(self):
        if self.current_id > 1:
            self.current_id -= 1
            self._load_lesson()

    def go_next(self):
        if self.current_id < len(self.outline):
            self.current_id += 1
            self._load_lesson()


# ============== 主窗口 ==============
class MainWindow(QMainWindow):
    status_signal = pyqtSignal(str, bool)

    def __init__(self):
        super().__init__()
        self.ollama_thread  = None
        self.rag_thread     = None
        self.ai_buffer      = ""
        self.thinking_buffer = ""
        self.current_ai_bubble = None
        self.current_thinking_bubble = None
        self.status_signal.connect(self._update_status)
        self.init_ui()
        self.check_ollama()

    def init_ui(self):
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(1000, 650)
        self.resize(1200, 750)
        self.setWindowIcon(self.create_icon())

        # 深色主题
        palette = QPalette()
        palette.setColor(QPalette.Window,      QColor(COLOR_BG))
        palette.setColor(QPalette.WindowText,  QColor(COLOR_TEXT))
        self.setPalette(palette)

        # 中央部件（含侧边栏）
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 左侧知识库边栏
        self.kb_sidebar = KnowledgeSidebar()
        self.kb_sidebar.setVisible(False)   # 默认隐藏
        main_layout.addWidget(self.kb_sidebar)

        # 右侧主区域
        right_area = QWidget()
        right_layout = QVBoxLayout(right_area)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # 工具栏
        self.create_toolbar()

        # 聊天区域
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        chat_layout.setContentsMargins(16, 16, 16, 16)
        chat_layout.setSpacing(12)

        welcome = QLabel(f"""<div style='text-align: center; padding: 40px;'>
            <h1 style='color: {COLOR_ACCENT}; margin: 0;'>🎓 学霸帝 AI v{APP_VERSION}</h1>
            <p style='color: {COLOR_TEXT_SECONDARY}; margin-top: 16px;'>
                本地 AI 导师 · 保护隐私 · Qwen3.5-2B<br>
                <span style='font-size: 12px;'>✨ 新增：RAG 知识库 + 引导式学习路径</span>
            </p>
            <p style='color: {COLOR_TEXT_SECONDARY}; margin-top: 24px; font-size: 13px;'>
                💬 输入问题，或上传文档启用知识库检索！
            </p>
        </div>""")
        chat_layout.addWidget(welcome)

        scroll = QScrollArea()
        scroll.setWidget(chat_widget)
        scroll.setStyleSheet(f"""
            QScrollArea {{ background-color: {COLOR_BG}; border: none; }}
            QScrollBar:vertical {{
                background: {COLOR_SURFACE}; width: 8px; border-radius: 4px;
            }}
            QScrollBar::handle {{ background: {COLOR_BORDER}; border-radius: 4px; }}
        """)
        scroll.setWidgetResizable(True)
        self.chat_widget = chat_widget
        self.chat_layout = chat_layout
        self.scroll_area = scroll

        right_layout.addWidget(scroll, 1)

        # 输入区域
        self.create_input_area(right_layout)

        # 状态栏
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; background: {COLOR_SURFACE};")
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("🟡 正在检查 Ollama 服务...")

        main_layout.addWidget(right_area, 1)

    def create_toolbar(self):
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setStyleSheet(f"""
            QToolBar {{
                background: {COLOR_SURFACE};
                border-bottom: 1px solid {COLOR_BORDER};
                padding: 8px;
            }}
        """)

        title_label = QLabel(f"<span style='color: {COLOR_ACCENT}; font-size: 16px; font-weight: bold;'>🎓 学霸帝 AI</span>")
        toolbar.addWidget(title_label)
        toolbar.addSeparator()

        # 知识库显隐
        kb_action = QAction("📚 知识库", self)
        kb_action.setCheckable(True)
        kb_action.toggled.connect(self.toggle_kb_sidebar)
        toolbar.addAction(kb_action)

        # 引导学习
        learn_action = QAction("📖 学习路径", self)
        learn_action.triggered.connect(self.open_learning_path)
        toolbar.addAction(learn_action)

        toolbar.addSeparator()

        refresh_action = QAction("🔄 检查服务", self)
        refresh_action.triggered.connect(self.check_ollama)
        toolbar.addAction(refresh_action)

        clear_action = QAction("🗑️ 清屏", self)
        clear_action.triggered.connect(self.clear_chat)
        toolbar.addAction(clear_action)

        self.model_label = QLabel(f"<span style='color: {COLOR_TEXT_SECONDARY};'>模型: {DEFAULT_MODEL}</span>")
        toolbar.addSeparator()
        toolbar.addWidget(self.model_label)

        self.addToolBar(toolbar)

    def toggle_kb_sidebar(self, checked):
        self.kb_sidebar.setVisible(checked)
        if checked:
            self.kb_sidebar.refresh_status()

    def open_learning_path(self):
        dlg = LearningPathDialog(self)
        dlg.exec_()

    def create_input_area(self, main_layout):
        input_frame = QFrame()
        input_frame.setStyleSheet(f"""
            QFrame {{
                background: {COLOR_SURFACE};
                border-top: 1px solid {COLOR_BORDER};
                padding: 12px;
            }}
        """)
        input_layout = QHBoxLayout(input_frame)
        input_layout.setSpacing(12)

        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("输入你的问题...（知识库命中时会自动参考）")
        self.input_box.setStyleSheet(f"""
            QLineEdit {{
                background: {COLOR_INPUT_BG};
                color: {COLOR_TEXT};
                border: 1px solid {COLOR_BORDER};
                border-radius: 20px;
                padding: 12px 20px;
                font-size: 14px;
            }}
            QLineEdit:focus {{ border: 1px solid {COLOR_ACCENT}; }}
            QLineEdit::placeholder {{ color: {COLOR_TEXT_SECONDARY}; }}
        """)
        self.input_box.setFont(QFont("Arial", 14))
        self.input_box.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_box, 1)

        self.send_btn = QPushButton("发送")
        self.send_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.send_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLOR_ACCENT};
                color: white;
                border: none;
                border-radius: 20px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background: #d63650; }}
            QPushButton:disabled {{ background: #555; }}
        """)
        self.send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_btn)

        main_layout.addWidget(input_frame)

    def create_icon(self):
        from PyQt5.QtGui import QPixmap, QPainter, QLinearGradient, QBrush, QPen
        pix = QPixmap(64, 64)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        grad = QLinearGradient(0, 0, 64, 64)
        grad.setColorAt(0, QColor(COLOR_ACCENT))
        grad.setColorAt(1, QColor("#c13350"))
        p.setBrush(QBrush(grad))
        p.setPen(Qt.NoPen)
        p.drawEllipse(4, 4, 56, 56)
        p.setPen(QPen(Qt.white, 3))
        p.drawLine(20, 22, 44, 22)
        p.drawLine(20, 32, 44, 32)
        p.drawLine(20, 42, 36, 42)
        p.end()
        return QIcon(pix)

    # ── Ollama 检查 ───────────────────────────────────────────────────────
    def check_ollama(self):
        self.status_bar.showMessage("🟡 正在检查 Ollama 服务...")
        threading.Thread(target=self._check_ollama_bg, daemon=True).start()

    def _check_ollama_bg(self):
        try:
            req = urllib.request.Request(f"{OLLAMA_API}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data   = json.loads(resp.read())
                models = [m["name"] for m in data.get("models", [])]
                self.status_signal.emit(
                    f"🟢 Ollama 运行中 | 模型: {', '.join(models) if models else '无'}",
                    True
                )
        except Exception:
            self.status_signal.emit("🔴 Ollama 未运行 | brew services start ollama", False)

    def _update_status(self, msg, ok):
        self.status_bar.showMessage(msg)
        if ok:
            self.model_label.setText(f"<span style='color: {COLOR_SUCCESS};'>{msg}</span>")

    # ── 发送消息 ─────────────────────────────────────────────────────────
    def send_message(self):
        text = self.input_box.text().strip()
        if not text:
            return

        if self.ollama_thread and self.ollama_thread.isRunning():
            QMessageBox.warning(self, "提示", "请等待上一次回答完成！")
            return

        self.input_box.clear()
        self.input_box.setEnabled(False)
        self.send_btn.setEnabled(False)

        self.add_message(text, is_user=True)

        self.ai_buffer       = ""
        self.thinking_buffer = ""
        self.current_ai_bubble = MessageBubble("", is_user=False)
        self.current_thinking_bubble = MessageBubble("", is_thinking=True)

        self.chat_layout.addWidget(self.current_ai_bubble)
        self.chat_layout.addWidget(self.current_thinking_bubble)
        self.current_thinking_bubble.hide()
        self.scroll_to_bottom()
        self.status_bar.showMessage("🤔 AI 正在思考...")

        # 构建 prompt（含 RAG）
        prompt = self._build_prompt(text)

        self.ollama_thread = OllamaStreamThread(prompt)
        self.ollama_thread.response_ready.connect(self._on_response)
        self.ollama_thread.thinking_ready.connect(self._on_thinking)
        self.ollama_thread.token_count.connect(self._on_token)
        self.ollama_thread.finished_signal.connect(self._on_finished)
        self.ollama_thread.start()

    def _build_prompt(self, text):
        # RAG 知识库检索
        try:
            results = rag_engine.search(text, top_k=5)
            if results:
                parts = ["以下是参考材料："]
                for i, r in enumerate(results, 1):
                    snippet = r["text"][:300].replace("\n", " ").strip()
                    parts.append(f"[{i}] 《{r['doc']}》：{snippet}")
                rag_context = "\n".join(parts) + "\n\n请根据以上参考材料回答问题。如果材料不足以回答，再结合你的知识回答。\n\n"
                self.status_bar.showMessage("📚 已启用知识库检索")
                return rag_context + text
        except Exception as e:
            print(f"[RAG] 检索失败: {e}", file=sys.stderr)

        return text

    # ── 流式响应处理 ────────────────────────────────────────────────────
    def _on_response(self, text):
        try:
            if not text:
                return
            self.ai_buffer += text
            if self.current_ai_bubble:
                self.current_ai_bubble.update_text(self.ai_buffer)
                self.scroll_to_bottom()
        except Exception as e:
            print(f"[_on_response ERROR] {e}", file=sys.stderr)

    def _on_thinking(self, text):
        try:
            if not text.strip():
                return
            self.thinking_buffer = text
            if self.current_thinking_bubble:
                self.current_thinking_bubble.show()
                self.current_thinking_bubble.update_text(text)
            self.scroll_to_bottom()
        except Exception as e:
            print(f"[_on_thinking ERROR] {e}", file=sys.stderr)

    def _on_token(self, count):
        try:
            self.status_bar.showMessage(f"🤔 思考中... ({count} tokens)")
        except Exception:
            pass

    def _on_finished(self, success, error_msg):
        try:
            self.input_box.setEnabled(True)
            self.send_btn.setEnabled(True)

            if not success:
                if self.current_ai_bubble:
                    self.current_ai_bubble.setParent(None)
                if self.current_thinking_bubble:
                    self.current_thinking_bubble.setParent(None)
                self.add_message(error_msg, is_user=False)
                self.status_bar.showMessage("❌ 推理失败")
            else:
                final = self.ai_buffer.strip()
                if not final:
                    final = "(模型未返回内容，请重试)"
                if self.current_ai_bubble:
                    self.current_ai_bubble.update_text(final)
                if self.current_thinking_bubble:
                    self.current_thinking_bubble.hide()
                self.status_bar.showMessage("✅ 回答完成")
            self.scroll_to_bottom()
        except Exception as e:
            print(f"[_on_finished ERROR] {e}", file=sys.stderr)
        finally:
            self.current_ai_bubble = None
            self.current_thinking_bubble = None

    # ── 聊天 UI 辅助 ────────────────────────────────────────────────────
    def add_message(self, text, is_user=False):
        bubble = MessageBubble(text, is_user=is_user)
        self.chat_layout.addWidget(bubble)
        self.scroll_to_bottom()

    def clear_chat(self):
        while self.chat_layout.count():
            item = self.chat_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        welcome = QLabel(f"""<div style='text-align: center; padding: 40px;'>
            <h1 style='color: {COLOR_ACCENT}; margin: 0;'>🎓 学霸帝 AI v{APP_VERSION}</h1>
            <p style='color: {COLOR_TEXT_SECONDARY}; margin-top: 16px;'>
                本地 AI 导师 · 保护隐私 · Qwen3.5-2B<br>
                <span style='font-size: 12px;'>✨ 新增：RAG 知识库 + 引导式学习路径</span>
            </p>
        </div>""")
        self.chat_layout.insertWidget(0, welcome)
        self.scroll_to_bottom()

    def scroll_to_bottom(self):
        QTimer.singleShot(50, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))


# ============== 入口 ==============
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    font = QFont("Arial", 13)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
