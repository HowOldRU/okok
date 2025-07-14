from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from elasticsearch import Elasticsearch
from uuid import uuid4
from config import ES_HOST, IMG_DIR
import shutil
import os

router = APIRouter(prefix="/doc", tags=["Document"])
es = Elasticsearch(ES_HOST, request_timeout=30)

def _fetch_chunks(kb_id: str, doc_id: str, full_content: bool = False) -> list[dict]:
    resp = es.search(
        index=kb_id,
        body={
            "query": {
                "bool": {
                    "must": [
                        {"term": {"doc_id": doc_id}}
                    ]
                }
            },
            "size": 1000  # 视实际块数量而定
        }
    )
    chunks = []
    for h in resp["hits"]["hits"]:
        source = h["_source"]
        chunk_type = source.get("type", "text")
        chunk_index = source.get("chunk_index", 0)

        if chunk_type == "image":
            content = source.get("content", "") or source.get("img_ocr", "")
            chunks.append({
                "chunk_id": source.get("chunk_id"),
                "chunk_index": chunk_index,
                "content": content if full_content else content[:200] + "..." if len(content) > 200 else content,
                "type": chunk_type,
                "title": source.get("title"),
                "date": source.get("date"),
                "image_path": source.get("image_path", ""),
                "img_caption": source.get("img_caption", ""),
                "img_ocr": source.get("img_ocr", ""),
            })
        elif chunk_type == "table":
            content = source.get("tbl_markdown", "")
            tbl_name = source.get("tbl_name", "")
            chunks.append({
                "chunk_id": source.get("chunk_id"),
                "chunk_index": chunk_index,
                "content": content if full_content else content[:200] + "..." if len(content) > 200 else content,
                "tbl_name": tbl_name,
                "type": chunk_type,
                "title": source.get("title"),
                "date": source.get("date"),
            })
        else:
            content = source.get("content", "")
            chunks.append({
                "chunk_id": source.get("chunk_id"),
                "chunk_index": chunk_index,
                "content": content if full_content else content[:200] + "..." if len(content) > 200 else content,
                "type": chunk_type,
                "title": source.get("title"),
                "date": source.get("date"),
            })

    # ★★★ 按 chunk_index 排序返回 ★★★
    return sorted(chunks, key=lambda x: x.get("chunk_index", 0))


@router.get("/list_docs")
def list_docs(kb_id: str):
    """
    获取指定知识库下所有文档的摘要信息。
    每个文档根据 doc_id 聚合，并提供部分内容片段作为预览。
    """

    if not es.indices.exists(index=kb_id):
        raise HTTPException(status_code=404, detail="知识库不存在")

    # 查出所有分块
    resp = es.search(
        index=kb_id,
        body={
            "size": 10000,
            "query": {"match_all": {}},
            "_source": [
                "doc_id", "title", "type", "date", "owner", "hits",
                "content", "img_caption", "img_ocr", "tbl_markdown"
            ]
        }
    )

    doc_map = {}
    for hit in resp["hits"]["hits"]:
        src = hit["_source"]
        doc_id = src["doc_id"]
        if doc_id not in doc_map:
            doc_map[doc_id] = {
                "doc_id": doc_id,
                "title": src.get("title", ""),
                "type": src.get("type", ""),
                "date": src.get("date"),
                "owner": src.get("owner", ""),
                "hits": src.get("hits", 0),
                "chunks": []
            }

        # 抽取摘要内容
        content = ""
        if src.get("type") == "text":
            content = src.get("content", "")
        elif src.get("type") == "image":
            content = src.get("img_caption") or src.get("img_ocr", "")
        elif src.get("type") == "table":
            content = src.get("tbl_markdown", "")
        if content:
            content = content.strip().replace("\n", " ")
            if len(content) > 200:
                content = content[:200] + "..."
            doc_map[doc_id]["chunks"].append(content)

    # 每个文档返回分块数量和摘要
    results = []
    for doc in doc_map.values():
        doc["chunk_count"] = len(doc["chunks"])
        doc["chunk_preview"] = doc.pop("chunks")
        results.append(doc)

    return results


@router.get("/{doc_id}")
def doc_detail(kb_id: str, doc_id: str):
    """获取指定文档的所有分块信息（按 doc_id 聚合）。"""
    if not es.indices.exists(index=kb_id):
        raise HTTPException(status_code=404, detail="知识库不存在")
    return {
        "doc_id": doc_id,
        "chunks": _fetch_chunks(kb_id, doc_id, full_content=True),
    }


@router.delete("/{doc_id}")
def delete_doc(kb_id: str, doc_id: str):
    """删除文档及其所有分块，若含图片则删除本地图片文件。"""
    if not es.indices.exists(index=kb_id):
        raise HTTPException(status_code=404, detail="知识库不存在")

    # 搜索所有分块
    resp = es.search(
        index=kb_id,
        body={
            "query": {
                "term": {"doc_id": doc_id}
            },
            "_source": ["image_path"],
            "size": 10000
        }
    )

    hits = resp["hits"]["hits"]

    # 删除 image 文件（从 image_path 中提取文件名）
    for h in hits:
        img_path = h["_source"].get("image_path")
        if img_path:
            fname = os.path.basename(img_path)
            fpath = os.path.join(IMG_DIR, fname)
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except Exception as e:
                    print(f"❌ 删除图片失败: {e}")

    # 删除所有分块文档
    for h in hits:
        es.delete(index=kb_id, id=h["_id"])

    return {"msg": f"文档 {doc_id} 删除成功", "chunks_deleted": len(hits)}
