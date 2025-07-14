"""search.py
增强版：三模态混合检索接口（支持检索后自动聚合前后文）
"""
from typing import List, Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from elasticsearch import Elasticsearch
import requests
from config import ES_HOST

EMBEDDING_URL = "http://10.146.138.192:11434/api/embeddings"
EMBEDDING_MODEL = "qwen3:8b"
ALPHA = 1
BETA_TEXT = 1
BETA_IMAGE = 1
BETA_TABLE = 1

router = APIRouter(prefix="/search", tags=["Search"])
es = Elasticsearch(ES_HOST, request_timeout=30)

class SearchRequest(BaseModel):
    kb_ids: List[str] = Field(..., description="一个或多个知识库索引名")
    query_text: str = Field(..., description="查询内容")
    top_k: int = Field(3, ge=1, le=100, description="返回结果条数")
    types: List[str] = Field(default=["text", "image", "table"], description="限制检索的块类型")

class Chunk(BaseModel):
    chunk_id: str | None = None
    chunk_index: int | None = None
    type: str | None = None
    content: str | None = None
    tbl_markdown: str | None = None
    tbl_name: str | None = None
    image_path: str | None = None
    img_caption: str | None = None
    img_ocr: str | None = None
    title: str | None = None
    doc_id: str | None = None
    date: str | None = None

class Document(BaseModel):
    text: str
    owner: str | None = None
    title: str | None = None
    date: str | None = None
    score: float | None = Field(None, description="融合得分")
    type: str | None = None
    image_url: str | None = None
    chunk_index: int | None = None
    doc_id: str | None = None
    context_chunks: List[Chunk] | None = None    # 新增，带完整上下文

def get_adjacent_chunks(es, index: str, doc_id: str, chunk_index: int, width: int = 3) -> List[dict]:
    """返回doc_id下chunk_index的前后块，按chunk_index排序"""
    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"doc_id": doc_id}},
                    {"range": {"chunk_index": {"gte": chunk_index-width, "lte": chunk_index+width}}}
                ]
            }
        },
        "size": 2*width + 1,
        "sort": [{"chunk_index": {"order": "asc"}}]
    }
    resp = es.search(index=index, body=query)
    return [h["_source"] for h in resp["hits"]["hits"]]

@router.post("/", response_model=List[Document])
def hybrid_search(req: SearchRequest):
    # === 1. 嵌入查询向量 ===
    try:
        emb_resp = requests.post(
            EMBEDDING_URL,
            json={"model": EMBEDDING_MODEL, "prompt": req.query_text},
            timeout=10
        )
        emb_resp.raise_for_status()
        query_vec = emb_resp.json()["embedding"]
    except Exception as exc:
        raise HTTPException(502, detail=f"Embedding error: {exc}")

    # === 2. BM25 检索（模态分字段）===
    bm25_query = {
        "size": req.top_k * 3,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": req.query_text,
                        "fields": [
                            "content^2",       # 文本
                            "title^2",  # 通用
                            "img_caption^2",   # 图片
                            "tbl_header^1",  # 表格内容
                            "tbl_name^2"       # 表格标题
                        ]
                    }
                },
                "filter": [
                    {"terms": {"type": req.types}}
                ]
            }
        }
    }
    try:
        bm25_res = es.search(index=req.kb_ids, body=bm25_query)
        bm25_hits = bm25_res["hits"]["hits"]
    except Exception:
        bm25_hits = []

    # === 3. 向量检索（严格分模态字段） ===
    knn_fields = [
        {"field": "vector_text", "weight": BETA_TEXT, "type": "text"},
        {"field": "vector_image", "weight": BETA_IMAGE, "type": "image"},
        {"field": "vector_table", "weight": BETA_TABLE, "type": "table"},
    ]
    knn_hits = []
    for kf in knn_fields:
        if kf["type"] not in req.types:
            continue
        try:
            qbody = {
                "size": req.top_k * 3,
                "knn": {
                    "field": kf["field"],
                    "query_vector": query_vec,
                    "k": req.top_k * 3,
                    "num_candidates": 100
                }
            }
            res = es.search(index=req.kb_ids, body=qbody)
            for h in res["hits"]["hits"]:
                h["_knn_field"] = kf["field"]
                h["_knn_weight"] = kf["weight"]
                knn_hits.append(h)
        except Exception:
            continue

    # === 4. 合并去重并融合打分 ===
    def extract_text(src: dict) -> str:
        t = src.get("type")
        if t == "image":
            return (src.get("content") or "img_ocr") + "\n" + (src.get("img_caption") or "")
        elif t == "table":
            name = src.get("tbl_name", "")
            header = src.get("tbl_header", "")
            return (name + "\n" if name else "") + header
        else:
            return src.get("content", "")

    result_pool: Dict[str, Dict] = {}
    def merge(hit: dict, is_bm25: bool, pool: dict):
        key = f"{hit['_index']}|{hit['_id']}"
        src = hit["_source"]
        if src.get("type") not in req.types:
            return
        entry = pool.setdefault(
            key,
            {
                "text": extract_text(src),
                "owner": src.get("owner"),
                "type": src.get("type"),
                "title": src.get("title"),
                "date": src.get("date"),
                "image_url": src.get("image_path") if src.get("type") == "image" else None,
                "chunk_index": src.get("chunk_index"),
                "doc_id": src.get("doc_id"),
                "bm25": 0.0,
                "vector": 0.0,
                "_index": hit["_index"],
                "_id": hit["_id"]
            }
        )
        if is_bm25:
            entry["bm25"] = hit["_score"]
        else:
            entry["vector"] += hit["_score"] * hit.get("_knn_weight", 1.0)

    for h in bm25_hits:
        merge(h, True, result_pool)
    for h in knn_hits:
        merge(h, False, result_pool)

    # === 5. 融合打分、更新 hits、并自动聚合上下文 ===
    docs: List[Document] = []
    # 遍历top命中的主块
    for info in sorted(result_pool.values(), key=lambda x: (-(x["bm25"] + x["vector"]))):
        fused = ALPHA * info["bm25"] + info["vector"]
        # 聚合上下文（前后1块）
        context_chunks = None
        if info.get("doc_id") and info.get("chunk_index") is not None:
            try:
                adj_chunks = get_adjacent_chunks(
                    es, info["_index"], info["doc_id"], info["chunk_index"], width=3)
                context_chunks = [
                    Chunk(
                        chunk_id=ch.get("chunk_id"),
                        chunk_index=ch.get("chunk_index"),
                        type=ch.get("type"),
                        content=ch.get("content"),
                        tbl_markdown=ch.get("tbl_markdown"),
                        tbl_name=ch.get("tbl_name"),
                        image_path=ch.get("image_path"),
                        img_caption=ch.get("img_caption"),
                        img_ocr=ch.get("img_ocr"),
                        title=ch.get("title"),
                        doc_id=ch.get("doc_id"),
                        date=ch.get("date"),
                    )
                    for ch in adj_chunks
                ]
            except Exception as e:
                print(f"❌ 上下文聚合失败: {e}")
        try:
            es.update(index=info["_index"], id=info["_id"], body={
                "script": {"source": "ctx._source.hits += 1", "lang": "painless"}
            })
        except Exception as e:
            print(f"❌ 更新 hits 失败（{info['_index']} | {info['_id']}）: {e}")
        docs.append(Document(
            text=info["text"],
            owner=info["owner"],
            title=info["title"],
            date=info["date"],
            score=fused,
            type=info["type"],
            image_url=info["image_url"],
            chunk_index=info.get("chunk_index"),
            doc_id=info.get("doc_id"),
            context_chunks=context_chunks
        ))

        if len(docs) >= req.top_k:
            break
    return docs
