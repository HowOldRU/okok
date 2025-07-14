from fastapi import APIRouter, HTTPException, Query
from elasticsearch import Elasticsearch
from config import ES_HOST, IMG_DIR
import os
from uuid import uuid4

router = APIRouter(prefix="/kb", tags=["KnowledgeBase"])
es = Elasticsearch(ES_HOST, request_timeout=30)


@router.post("/create")
def create_kb(kb_id: str = Query(..., description="知识库唯一标识符")):
    """
    根据kb_id创建一个新的知识库索引。
    """

    # kb_id = str(uuid4())
    if es.indices.exists(index=kb_id):
        raise HTTPException(status_code=400, detail=f"知识库 {kb_id} 已存在")

    # 映射字段结构
    mapping = {
        "mappings": {
            "properties": {
                "kb_id":    {"type": "keyword"},
                "doc_id":   {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                "type":     {"type": "keyword"},  # text | image | table

                "title": {"type": "text"},
                "date":  {"type": "date"},
                "owner": {"type": "keyword"},
                "hits":  {"type": "integer"},

                "content":   {"type": "text"},
                "keywords":  {"type": "keyword"},
                "vector_text": {
                    "type": "dense_vector",
                    "dims": 4096,
                    "index": True,
                    "similarity": "cosine"
                },

                "image_path":  {"type": "keyword"},
                "img_caption": {"type": "text"},
                "img_ocr":     {"type": "text"},
                "vector_image": {
                    "type": "dense_vector",
                    "dims": 4096,
                    "index": True,
                    "similarity": "cosine"
                },

                "tbl_markdown": {"type": "text"},
                "tbl_header":   {"type": "keyword"},
                "tbl_name": {"type": "keyword"},
                "vector_table": {
                    "type": "dense_vector",
                    "dims": 4096,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }

    es.indices.create(index=kb_id, body=mapping)
    return {"msg": f"知识库 {kb_id} 创建成功"}


@router.delete("/{kb_id}")
def delete_kb(kb_id: str):
    """
    删除指定的知识库索引及其关联图片资源。
    """

    if not es.indices.exists(index=kb_id):
        raise HTTPException(status_code=404, detail="知识库不存在")

    # 查出所有 image_path 字段
    resp = es.search(
        index=kb_id,
        body={
            "query": {
                "exists": {"field": "image_path"}
            },
            "_source": ["image_path"],
            "size": 10000
        }
    )

    # 删除所有相关图片文件
    for h in resp["hits"]["hits"]:
        image_path = h["_source"].get("image_path")
        if image_path:
            fname = os.path.basename(image_path)
            fpath = IMG_DIR / fname
            if fpath.exists():
                try:
                    fpath.unlink()
                except Exception as e:
                    print(f"❌ 删除图片失败: {e}")

    # 删除整个索引
    es.indices.delete(index=kb_id)
    return {"msg": f"知识库 {kb_id} 删除成功"}


@router.get("/list")
def list_all_kbs():
    """
    获取当前所有知识库索引，返回每个库的统计信息。

    Returns: 包含所有 kb_id 的文档数、chunk 数、最新上传时间
    """
    # 排除系统索引（以.开头）
    all_indices = list(es.indices.get_alias().keys())
    kb_ids = [idx for idx in all_indices if not idx.startswith(".")]

    results = []

    for kb_id in kb_ids:
        try:
            # 统计文档总数（distinct doc_id）
            doc_aggs = es.search(
                index=kb_id,
                size=0,
                aggs={
                    "distinct_docs": {
                        "cardinality": { "field": "doc_id" }
                    },
                    "latest_date": {
                        "max": { "field": "date" }
                    }
                }
            )

            doc_count = doc_aggs["aggregations"]["distinct_docs"]["value"]
            latest_date = doc_aggs["aggregations"]["latest_date"].get("value_as_string", "N/A")
            chunk_count = doc_aggs["hits"]["total"]["value"]

            results.append({
                "kb_id": kb_id,
                "doc_count": doc_count,
                "chunk_count": chunk_count,
                "latest_date": latest_date
            })

        except Exception as e:
            results.append({
                "kb_id": kb_id,
                "doc_count": "N/A",
                "chunk_count": "N/A",
                "latest_date": "N/A",
                "error": str(e)
            })

    return {"knowledge_bases": results}
