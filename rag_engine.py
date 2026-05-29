"""
rag_engine.py — 学霸帝 AI 本地 RAG 知识库引擎
方案：Ollama nomic-embed-text 做 embedding + FAISS 做向量检索
零 torch 依赖，Intel Mac 友好
"""

import os
import json
import glob
import urllib.request
import urllib.error
import numpy as np
import faiss
import fitz  # PyMuPDF

# ── 配置 ────────────────────────────────────────────────────────────────────
OLLAMA_API    = "http://localhost:11434"
EMBED_MODEL   = "nomic-embed-text"   # ollama pull nomic-embed-text
CHUNK_SIZE    = 500    # 每片字符数
CHUNK_OVERLAP = 80     # 片重叠字符数
TOP_K         = 5

INDEX_PATH    = os.path.expanduser("~/.xuebadi_ai/kb/index.faiss")
META_PATH     = os.path.expanduser("~/.xuebadi_ai/kb/meta.json")
DOC_DIR       = os.path.expanduser("~/.xuebadi_ai/kb/docs/")

os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
os.makedirs(DOC_DIR, exist_ok=True)

_index = None
_meta  = []   # [{"doc": "xxx.pdf", "text": "..."}]


# ── Embedding（调用 Ollama）────────────────────────────────────────────────
def _ensure_embedding_model() -> bool:
    """检查 nomic-embed-text 是否已拉取"""
    try:
        req = urllib.request.Request(
            f"{OLLAMA_API}/api/show",
            data=json.dumps({"name": EMBED_MODEL}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def _ollama_embed(texts: list[str]) -> np.ndarray:
    url = f"{OLLAMA_API}/api/embeddings"
    all_vecs = []
    for text in texts:
        payload = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode()
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        all_vecs.append(data["embedding"])
    return np.array(all_vecs, dtype="float32")


def _get_dim() -> int:
    vec = _ollama_embed(["探测"])[0]
    return len(vec)


# ── FAISS 索引管理 ────────────────────────────────────────────────────────
def _get_index():
    global _index, _meta
    if _index is not None:
        return _index, _meta
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        print("[RAG] 加载已有向量索引...")
        _index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            _meta = json.load(f)
        print(f"[RAG] 索引已加载，共 {_index.ntotal} 条")
    else:
        # 延迟探测维度，避免导入时就报错
        try:
            dim = _get_dim()
        except Exception as e:
            print(f"[RAG] ⚠️ 无法探测 embedding 维度: {e}")
            print("[RAG] ⚠️ 请先运行: ollama pull nomic-embed-text")
            dim = 768  # nomic-embed-text 默认维度，先占位
        _index = faiss.IndexFlatL2(dim)
        _meta = []
        print(f"[RAG] 新建索引 (dim={dim})")
    return _index, _meta


def _save():
    global _index, _meta
    faiss.write_index(_index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(_meta, f, ensure_ascii=False, indent=2)
    print(f"[RAG] 索引已保存 ({_index.ntotal} 条)")


# ── 文档解析 ────────────────────────────────────────────────────────────────
def _parse_pdf(path: str) -> str:
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


def _parse_md(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _parse_txt(path: str) -> str:
    for enc in ("utf-8", "gbk", "gb18030", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    return ""


def parse_document(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _parse_pdf(path)
    elif ext in (".md", ".markdown"):
        return _parse_md(path)
    else:
        return _parse_txt(path)


# ── 分块 ───────────────────────────────────────────────────────────────────
def _chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + CHUNK_SIZE, text_len)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
        if start <= 0:
            break
    return chunks


# ── 索引构建 ───────────────────────────────────────────────────────────────
def add_document(path: str) -> int:
    """解析文档 → 分块 → embedding → FAISS，返回新增 chunk 数"""
    global _index, _meta
    index, meta = _get_index()

    base = os.path.basename(path)
    # 去重：已索引过的文档跳过
    if any(m["doc"] == base for m in meta):
        print(f"[RAG] 跳过已索引: {base}")
        return 0

    print(f"[RAG] 解析: {base}")
    text = parse_document(path)
    if not text.strip():
        print("[RAG] 警告: 文档内容为空")
        return 0

    chunks = _chunk_text(text)
    print(f"[RAG] 分块数: {len(chunks)}，正在编码...")

    vecs = _ollama_embed(chunks)   # shape=(N, dim)
    index.add(vecs)

    for ch in chunks:
        meta.append({"doc": base, "text": ch})

    _save()
    return len(chunks)


def add_documents_from_dir(doc_dir: str = DOC_DIR) -> int:
    total = 0
    for ext in ("*.pdf", "*.md", "*.txt"):
        for path in sorted(glob.glob(os.path.join(doc_dir, ext))):
            total += add_document(path)
    return total


# ── 检索 ───────────────────────────────────────────────────────────────────
def search(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    返回 [{"text":..., "doc":..., "score":...}, ...]
    score 是 L2 距离，越小越相关
    """
    index, meta = _get_index()
    if index.ntotal == 0:
        return []

    q_vec = _ollama_embed([query])   # shape=(1, dim)
    distances, indices = index.search(q_vec, min(top_k, index.ntotal))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(meta):
            continue
        results.append({
            "text":  meta[idx]["text"],
            "doc":   meta[idx]["doc"],
            "score": float(dist),
        })
    return results


def search_formatted(query: str, top_k: int = TOP_K) -> str:
    """返回可直接拼入 Ollama prompt 的参考文本"""
    results = search(query, top_k)
    if not results:
        return ""
    lines = ["【参考材料】"]
    for i, r in enumerate(results, 1):
        snippet = r["text"][:400].replace("\n", " ").strip()
        lines.append(f"{i}. 《{r['doc']}》：{snippet}")
    return "\n".join(lines)


# ── 知识库状态 & 管理 ────────────────────────────────────────────────────
def kb_status() -> dict:
    """返回知识库状态，embedding 模型未就绪时也可安全调用"""
    index, meta = _get_index()
    docs = {}
    for m in meta:
        d = m["doc"]
        docs[d] = docs.get(d, 0) + 1
    ready = _ensure_embedding_model()
    return {
        "total_chunks": int(index.ntotal),
        "doc_count":   len(docs),
        "docs":        docs,
        "doc_dir":     DOC_DIR,
        "index_path":  INDEX_PATH,
        "embed_ready": ready,
    }


def remove_document(doc_name: str) -> bool:
    global _index, _meta
    index, meta = _get_index()
    keep = [m for m in meta if m["doc"] != doc_name]
    if len(keep) == len(meta):
        return False

    print(f"[RAG] 重建索引（删除 {doc_name}）...")
    texts = [m["text"] for m in keep]
    dim = _index.d   # 用已有索引的维度，避免重新探测
    _index = faiss.IndexFlatL2(dim)
    if texts:
        vecs = _ollama_embed(texts)
        _index.add(vecs)
    _meta = keep
    _save()
    print(f"[RAG] 完成，剩余 {_index.ntotal} 条")
    return True


def clear_kb() -> None:
    global _index, _meta
    dim = _index.d if _index is not None else 768
    _index = faiss.IndexFlatL2(dim)
    _meta = []
    _save()
    print("[RAG] 知识库已清空")
