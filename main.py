"""
学霸帝 AI - macOS 本地导师应用
基于 PyQt5 + Ollama Qwen3.5-2B 本地模型
"""
import sys
import json
import threading
import urllib.request
import urllib.error
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QLineEdit, QScrollArea,
    QFrame, QSplitter, QStatusBar, QComboBox, QAction, QToolBar,
    QMessageBox, QGraphicsDropShadowEffect
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon, QTextCursor, QTextCharFormat, QCursor


# ============== 配置 ==============
OLLAMA_API = "http://localhost:11434"
DEFAULT_MODEL = "xueba-di"
APP_NAME = "学霸帝 AI"
APP_VERSION = "1.0.0"

# ============== 颜色主题 ==============
COLOR_BG = "#1a1a2e"
COLOR_SURFACE = "#16213e"
COLOR_USER_BG = "#0f3460"
COLOR_AI_BG = "#1e3a5f"
COLOR_ACCENT = "#e94560"
COLOR_TEXT = "#eaeaea"
COLOR_TEXT_SECONDARY = "#8892b0"
COLOR_BORDER = "#2a3f5f"
COLOR_INPUT_BG = "#0f3460"


# ============== 流式响应线程 ==============
class OllamaStreamThread(QThread):
    response_ready = pyqtSignal(str)
    thinking_ready = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    token_count = pyqtSignal(int)

    def __init__(self, prompt, model=DEFAULT_MODEL, parent=None):
        super().__init__(parent)
        self.prompt = prompt
        self.model = model
        self._running = True

    def run(self):
        try:
            # 检查模型是否加载
            try:
                req = urllib.request.Request(
                    f"{OLLAMA_API}/api/show",
                    data=json.dumps({"name": self.model}).encode(),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    pass
            except Exception:
                self.finished_signal.emit(False, f"❌ 模型 '{self.model}' 未找到。请先在终端运行：ollama create xueba-di")
                return

            # 发送请求
            req = urllib.request.Request(
                f"{OLLAMA_API}/api/generate",
                data=json.dumps({
                    "model": self.model,
                    "prompt": self.prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 512,
                    }
                }).encode(),
                headers={"Content-Type": "application/json"}
            )

            buffer = ""
            token_count = 0
            thinking_mode = False
            thinking_text = ""

            with urllib.request.urlopen(req, timeout=300) as resp:
                for line in resp:
                    if not self._running:
                        break
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        buffer += token
                        token_count += 1

                        # 检测思考模式
                        if "<think>" in buffer:
                            thinking_mode = True
                            before_think = buffer.split("<think>")[0]
                            if before_think:
                                self.response_ready.emit(before_think)
                                buffer = "<think>" + buffer.split("<think>")[1]
                        elif "</think>" in buffer and thinking_mode:
                            thinking_mode = False
                            thinking_content = buffer.split("</think>")[0].replace("<think>", "")
                            self.thinking_ready.emit(thinking_content)
                            buffer = buffer.split("</think>")[1]

                        if not thinking_mode:
                            self.response_ready.emit(token)
                        else:
                            # 更新思考内容
                            current_think = buffer.replace("<think>", "").replace("</think>", "")
                            self.thinking_ready.emit(current_think)

                        self.token_count.emit(token_count)

                    except json.JSONDecodeError:
                        continue

            self.finished_signal.emit(True, "")

        except urllib.error.URLError as e:
            self.finished_signal.emit(False, f"❌ 连接失败：Ollama 服务未启动\n\n请在终端运行：brew services start ollama\n或者下载 Ollama: https://ollama.ai")
        except Exception as e:
            self.finished_signal.emit(False, f"❌ 错误：{str(e)}")

    def stop(self):
        self._running = False


# ============== 消息气泡 ==============
class MessageBubble(QFrame):
    def __init__(self, text, is_user=False, is_thinking=False, parent=None):
        super().__init__(parent)
        self.is_user = is_user
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


# ============== 主窗口 ==============
class MainWindow(QMainWindow):
    status_signal = pyqtSignal(str, bool)

    def __init__(self):
        super().__init__()
        self.ollama_thread = None
        self.ai_buffer = ""
        self.thinking_buffer = ""
        self.current_ai_bubble = None
        self.current_thinking_bubble = None
        self.status_signal.connect(self._update_status)
        self.init_ui()
        self.check_ollama()

    def init_ui(self):
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(800, 600)
        self.resize(1000, 700)
        self.setWindowIcon(self.create_icon())

        # 设置深色主题
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(COLOR_BG))
        palette.setColor(QPalette.WindowText, QColor(COLOR_TEXT))
        self.setPalette(palette)

        # 中央部件
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 工具栏
        self.create_toolbar()

        # 聊天区域
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        chat_layout.setContentsMargins(16, 16, 16, 16)
        chat_layout.setSpacing(12)

        # 欢迎消息
        welcome = QLabel(f"""<div style='text-align: center; padding: 40px;'>
            <h1 style='color: {COLOR_ACCENT}; margin: 0;'>🎓 学霸帝 AI</h1>
            <p style='color: {COLOR_TEXT_SECONDARY}; margin-top: 16px;'>
                本地 AI 导师 · 保护隐私 · Qwen3.5-2B 模型<br>
                <span style='font-size: 12px;'>基于 Ollama 推理引擎</span>
            </p>
            <p style='color: {COLOR_TEXT_SECONDARY}; margin-top: 24px; font-size: 13px;'>
                💬 输入你的问题，AI 导师会为你解答！
            </p>
        </div>""")
        chat_layout.addWidget(welcome)

        # 滚动区域
        scroll = QScrollArea()
        scroll.setWidget(chat_widget)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {COLOR_BG};
                border: none;
            }}
            QScrollBar:vertical {{
                background: {COLOR_SURFACE};
                width: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle {{
                background: {COLOR_BORDER};
                border-radius: 4px;
            }}
        """)
        scroll.setWidgetResizable(True)
        self.chat_widget = chat_widget
        self.chat_layout = chat_layout
        self.scroll_area = scroll

        main_layout.addWidget(scroll, 1)

        # 输入区域
        self.create_input_area(main_layout)

        # 状态栏
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; background: {COLOR_SURFACE};")
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("🟡 正在检查 Ollama 服务...")

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

        refresh_action = QAction("🔄 检查服务", self)
        refresh_action.triggered.connect(self.check_ollama)
        toolbar.addAction(refresh_action)

        clear_action = QAction("🗑️ 清空对话", self)
        clear_action.triggered.connect(self.clear_chat)
        toolbar.addAction(clear_action)

        self.model_label = QLabel(f"<span style='color: {COLOR_TEXT_SECONDARY};'>模型: {DEFAULT_MODEL}</span>")
        toolbar.addSeparator()
        toolbar.addWidget(self.model_label)

        self.addToolBar(toolbar)

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
        self.input_box.setPlaceholderText("输入你的问题...")
        self.input_box.setStyleSheet(f"""
            QLineEdit {{
                background: {COLOR_INPUT_BG};
                color: {COLOR_TEXT};
                border: 1px solid {COLOR_BORDER};
                border-radius: 20px;
                padding: 12px 20px;
                font-size: 14px;
            }}
            QLineEdit:focus {{
                border: 1px solid {COLOR_ACCENT};
            }}
            QLineEdit::placeholder {{
                color: {COLOR_TEXT_SECONDARY};
            }}
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
            QPushButton:hover {{
                background: #d63650;
            }}
            QPushButton:disabled {{
                background: #555;
            }}
        """)
        self.send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_btn)

        main_layout.addWidget(input_frame)

    def create_icon(self):
        # 创建一个简单的图标
        from PyQt5.QtGui import QPixmap, QPainter, QLinearGradient, QBrush, QPen, QIcon
        pix = QPixmap(64, 64)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        # 渐变背景
        grad = QLinearGradient(0, 0, 64, 64)
        grad.setColorAt(0, QColor(COLOR_ACCENT))
        grad.setColorAt(1, QColor("#c13350"))
        p.setBrush(QBrush(grad))
        p.setPen(Qt.NoPen)
        p.drawEllipse(4, 4, 56, 56)
        # 书本符号
        p.setPen(QPen(Qt.white, 3))
        p.drawLine(20, 22, 44, 22)
        p.drawLine(20, 32, 44, 32)
        p.drawLine(20, 42, 36, 42)
        p.end()
        return QIcon(pix)

    def check_ollama(self):
        self.status_bar.showMessage("🟡 正在检查 Ollama 服务...")
        threading.Thread(target=self._check_ollama_bg, daemon=True).start()

    def _check_ollama_bg(self):
        try:
            req = urllib.request.Request(f"{OLLAMA_API}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                models = [m["name"] for m in data.get("models", [])]
                self.status_signal.emit(
                    f"🟢 Ollama 运行中 | 可用模型: {', '.join(models) if models else '无'}",
                    True
                )
        except Exception:
            self.status_signal.emit(
                "🔴 Ollama 未运行 | 请运行: brew services start ollama",
                False
            )

    def _update_status(self, msg, ok):
        self.status_bar.showMessage(msg)
        if ok:
            self.model_label.setText(f"<span style='color: #4ade80;'>🟢 {msg}</span>")

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

        # 添加用户消息
        self.add_message(text, is_user=True)

        # 准备 AI 消息占位
        self.ai_buffer = ""
        self.thinking_buffer = ""
        self.current_ai_bubble = MessageBubble("", is_user=False)
        self.current_thinking_bubble = MessageBubble("", is_thinking=True)

        # 按顺序添加：先 AI 气泡，再思考气泡
        self.chat_layout.addWidget(self.current_ai_bubble)
        self.chat_layout.addWidget(self.current_thinking_bubble)
        self.current_thinking_bubble.hide()  # 初始隐藏

        self.scroll_to_bottom()
        self.status_bar.showMessage("🤔 AI 正在思考...")

        # 启动推理线程
        prompt = self._build_prompt(text)
        self.ollama_thread = OllamaStreamThread(prompt)
        self.ollama_thread.response_ready.connect(self._on_response)
        self.ollama_thread.thinking_ready.connect(self._on_thinking)
        self.ollama_thread.token_count.connect(self._on_token)
        self.ollama_thread.finished_signal.connect(self._on_finished)
        self.ollama_thread.start()

    def _build_prompt(self, text):
        return f"""<|im_start|>user
你叫学霸帝AI，是一位友善、有耐心的AI导师。你的特点是：
1. 回答清晰有条理，善于用例子解释
2. 鼓励用户思考，引导而非直接给答案
3. 回答简洁但完整，避免冗长
4. 遇到不懂的问题会坦诚说明

用户问题：{text}
<|im_end|>
<|im_start|>assistant"""

    def _on_response(self, text):
        if text == "":
            return
        self.ai_buffer += text
        if self.current_ai_bubble:
            # 实时显示：只显示 </think> 之后的内容（即正式回答）
            display = self._strip_think(self.ai_buffer)
            self.current_ai_bubble.update_text(display)

    def _on_thinking(self, text):
        if not text.strip():
            return
        self.thinking_buffer = text
        if self.current_thinking_bubble:
            self.current_thinking_bubble.show()
            self.current_thinking_bubble.update_text(text)
        self.scroll_to_bottom()

    def _strip_think(self, text):
        """移除 <think>...</think> 标签，只返回正式回答"""
        result = []
        i = 0
        while i < len(text):
            start = text.find("<think>", i)
            if start == -1:
                result.append(text[i:])
                break
            result.append(text[i:start])
            end = text.find("</think>", start)
            if end == -1:
                # 思考未结束，跳过 <think> 内容
                break
            i = end + len("</think>")
        return "".join(result).strip()

    def _on_token(self, count):
        self.status_bar.showMessage(f"🤔 AI 思考中... ({count} tokens)")

    def _on_finished(self, success, error_msg):
        self.input_box.setEnabled(True)
        self.send_btn.setEnabled(True)

        if not success:
            # 移除失败的占位
            if self.current_ai_bubble:
                self.current_ai_bubble.setParent(None)
            if self.current_thinking_bubble:
                self.current_thinking_bubble.setParent(None)
            self.add_message(error_msg, is_user=False)
            self.status_bar.showMessage("❌ 推理失败")
        else:
            # 最终显示（移除思考标签）
            final = self._strip_think(self.ai_buffer)
            if not final:
                final = "(模型未返回回答内容，请重试)"

            if self.current_ai_bubble:
                self.current_ai_bubble.update_text(final)
            if self.current_thinking_bubble:
                self.current_thinking_bubble.hide()
            self.status_bar.showMessage("✅ 回答完成")

        self.scroll_to_bottom()
        self.current_ai_bubble = None
        self.current_thinking_bubble = None

    def add_message(self, text, is_user=False):
        bubble = MessageBubble(text, is_user=is_user)
        self.chat_layout.addWidget(bubble)
        self.scroll_to_bottom()

    def clear_chat(self):
        # 移除所有消息气泡
        while self.chat_layout.count():
            item = self.chat_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        # 重新添加欢迎
        welcome = QLabel(f"""<div style='text-align: center; padding: 40px;'>
            <h1 style='color: {COLOR_ACCENT}; margin: 0;'>🎓 学霸帝 AI</h1>
            <p style='color: {COLOR_TEXT_SECONDARY}; margin-top: 16px;'>
                本地 AI 导师 · 保护隐私 · Qwen3.5-2B 模型<br>
                <span style='font-size: 12px;'>基于 Ollama 推理引擎</span>
            </p>
        </div>""")
        self.chat_layout.insertWidget(0, welcome)
        self.scroll_to_bottom()

    def scroll_to_bottom(self):
        QTimer.singleShot(50, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))





if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)


    font = QFont("Arial", 13)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
