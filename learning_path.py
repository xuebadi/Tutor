"""
learning_path.py — 学霸帝 AI 引导式学习路径引擎
功能：
  1. 解析学习材料 → 提取知识点大纲（调用本地 Ollama）
  2. 生成每个知识点的交互式讲解页面（流式）
  3. 跟踪学习进度，支持「上一步 / 下一步 / 跳到...」
"""

import json
import urllib.request
import urllib.error
import re

OLLAMA_API = "http://localhost:11434"
DEFAULT_MODEL = "xueba-di"


# ── 调用 Ollama（非流式，等完整回复）────────────────────────────────────
def _ollama_generate(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.3) -> str:
    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream":  False,
        "options": {
            "temperature": temperature,
            "num_predict": 1024,
            "num_ctx": 2048,
        }
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_API}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    return data.get("response", "")


# ── 1. 从学习材料提取知识点大纲 ────────────────────────────────────────
def extract_outline(material_text: str, model: str = DEFAULT_MODEL) -> list[dict]:
    """
    输入：学习材料全文
    输出：[{ "id": 1, "title": "...", "summary": "..." }, ...]
    """
    prompt = f"""你是一位专业的课程设计师。请仔细分析下面这段学习材料，
用中文拆解出知识点大纲。

要求：
- 每个知识点一行，格式：编号|标题|20字内摘要
- 编号从1开始连续
- 只输出大纲，不要解释

示例输出：
1|什么是变量|理解变量的定义和用途
2|变量的命名规则|掌握命名规范
3|变量的数据类型|了解常见数据类型

学习材料：
{material_text[:3000]}
"""
    raw = _ollama_generate(prompt, model)
    outline = []
    for line in raw.strip().split("\n"):
        m = re.match(r"(\d+)\|(.+?)\|(.+)", line.strip())
        if m:
            outline.append({
                "id":      int(m.group(1)),
                "title":   m.group(2).strip(),
                "summary": m.group(3).strip(),
            })
    if not outline:
        # fallback：按行强行解析
        for i, l in enumerate(raw.strip().split("\n")[:20], 1):
            t = re.sub(r"^\d+[\.、\)\]]\s*", "", l.strip())
            if t:
                outline.append({"id": i, "title": t[:40], "summary": ""})
    return outline


# ── 2. 生成单个知识点的交互式讲解 ──────────────────────────────────────
def generate_lesson(outline: list[dict], current_id: int, material_text: str = "",
                    model: str = DEFAULT_MODEL) -> dict:
    """
    输入：大纲 + 当前知识点 id + 原始材料
    输出：{
        "title":   "...",
        "content": "讲解正文（含例题）",
        "quiz":    "随堂小测（2-3题）",
        "next_id": int or None,
        "prev_id": int or None,
    }
    """
    item = next((o for o in outline if o["id"] == current_id), None)
    if not item:
        return {"error": f"未找到 id={current_id} 的知识点"}

    prev_id = current_id - 1 if current_id > 1 else None
    next_id = current_id + 1 if current_id < len(outline) else None

    # ── 生成讲解正文 ──────────────────────────────────────────────────
    prompt_content = f"""你是一位耐心的AI导师，正在讲解「{item['title']}」这个知识点。

要求：
1. 用中文讲解，语言生动，举生活例子
2. 分 2-3 个小节，每节用 ## 标题
3. 结尾出 1 道选择题（附答案，答案用 ## 答案 折叠）
4. 总长度 300-500 字，不要啰嗦
5. 不要输出「参考答案」以外的内容

请直接输出讲解内容：
"""
    content = _ollama_generate(prompt_content, model, temperature=0.7)

    return {
        "title":    item["title"],
        "summary":  item["summary"],
        "content":  content,
        "prev_id":  prev_id,
        "next_id":  next_id,
        "total":    len(outline),
        "current":  current_id,
    }


# ── 3. 流式生成讲解（用于 UI 渐进显示）────────────────────────────────
def generate_lesson_stream(outline: list[dict], current_id: int,
                           model: str = DEFAULT_MODEL):
    """
    生成器，逐 token yield 讲解内容。
    用法：
        for token in generate_lesson_stream(outline, 1):
            ...
    """
    item = next((o for o in outline if o["id"] == current_id), None)
    if not item:
        yield json.dumps({"error": f"未找到 id={current_id}"})
        return

    prompt = f"""你是一位耐心的AI导师，正在讲解「{item['title']}」。

要求：
1. 用中文讲解，语言生动，举生活例子
2. 分 2-3 个小节，每节用 ## 标题
3. 结尾出 1 道选择题（答案附在 ## 答案 后面，折叠显示）
4. 总长度 300-500 字

直接输出讲解内容："""
    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.7,
            "num_predict": 1024,
            "num_ctx": 2048,
        }
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_API}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        for line in resp:
            try:
                data = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield token
            except Exception:
                continue


# ── 4. 问答检测（判断用户是否真正掌握了）────────────────────────────
def check_answer(question: str, user_answer: str,
                 model: str = DEFAULT_MODEL) -> dict:
    """
    返回 {"correct": bool, "explanation": "..."}
    """
    prompt = f"""你是一位导师，判断学生的回答是否正确。

题目：{question}
学生回答：{user_answer}

请按以下格式回答（只输出 JSON，不要多余内容）：
{{"correct": true/false, "explanation": "一句话解释"}}"""
    raw = _ollama_generate(prompt, model, temperature=0.1)
    # 尝试提取 JSON
    try:
        j = json.loads(raw[raw.index("{"):raw.index("}")+1])
        return j
    except Exception:
        # fallback
        correct = "正确" in raw or "✓" in raw or "对" in raw[:20]
        return {"correct": correct, "explanation": raw.strip()[:200]}


# ── 5. 生成完整学习路径报告（一次性）─────────────────────────────────
def generate_path_report(material_text: str, model: str = DEFAULT_MODEL) -> str:
    """
    输入学习材料，输出完整 Markdown 学习路径报告
    （用于「导出学习路径」功能）
    """
    prompt = f"""你是一位课程设计师。根据下面的学习材料，
生成一份结构化的学习路径报告（Markdown 格式）。

报告结构：
# 学习路径：xxx
## 学习目标
## 知识点大纲（含每个知识点的预计学习时间）
## 学习建议（3-5 条）
## 推荐练习

学习材料：
{material_text[:4000]}
"""
    return _ollama_generate(prompt, model, temperature=0.3)


# ── 6. 解析用户上传的文件，提取纯文本 ────────────────────────────────
def extract_text_from_file(path: str) -> str:
    """支持 PDF / MD / TXT，返回纯文本"""
    import fitz
    ext = __import__("os").path.splitext(path)[1].lower()
    if ext == ".pdf":
        doc = fitz.open(path)
        return "\n".join(p.get_text() for p in doc)
    elif ext in (".md", ".markdown"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        for enc in ("utf-8", "gbk", "gb18030"):
            try:
                with open(path, "r", encoding=enc) as f:
                    return f.read()
            except Exception:
                continue
        return ""
