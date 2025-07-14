import os, re, uuid, base64, requests, zipfile
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from typing import Literal
from elasticsearch import Elasticsearch, helpers
from elasticsearch.helpers import BulkIndexError
from collections import defaultdict
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from config import ES_HOST
from docx import Document
from docx.document import Document as _Document
from docx.table import Table
from docx.text.paragraph import Paragraph
import pandas as pd
from io import BytesIO

EMBEDDING_URL   = "http://10.146.138.192:11434/api/embeddings"
EMBEDDING_MODEL = "qwen3:8b"
VLLM_URL        = "http://10.199.194.246:3002/v1/chat/completions"
VLLM_MODEL_ID   = "/models/Qwen2.5-VL-7B-Instruct"

es = Elasticsearch(ES_HOST, request_timeout=30)
router = APIRouter(prefix="/doc", tags=["Document"])
STATIC_IMG_DIR = "/home/okok/nginx-static/static/img1"
IMG_BASE_URL = "http://10.199.194.246:8081/img1"

def embed(text: str, expected_dim=4096):
    if not text:
        return None
    try:
        prompt = f"请生成以下文本的语义向量表示：\n{text}"
        response = requests.post(
            EMBEDDING_URL,
            json={"model": EMBEDDING_MODEL, "prompt": prompt},
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        embedding = result.get("embedding", [])
        if len(embedding) != expected_dim:
            raise ValueError(f"向量维度错误：期望 {expected_dim}，实际 {len(embedding)}")
        return embedding
    except Exception as e:
        print(f"获取嵌入向量失败: {e}")
        return None

def generate_caption_from_vllm(image_bytes: bytes) -> str:
    base64_image = base64.b64encode(image_bytes).decode()
    image_data_url = f"data:image/jpeg;base64,{base64_image}"
    payload = {
        "model": VLLM_MODEL_ID,
        "messages": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": "请描述这张图片"}
            ]}
        ],
        "temperature": 0.7,
        "max_tokens": 128
    }
    try:
        response = requests.post(VLLM_URL, headers={"Content-Type": "application/json"}, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[caption 生成失败] {e}")
        return ""

def iter_block_items(doc):
    body = doc.element.body
    for child in body.iterchildren():
        if child.tag.endswith('}p'):
            yield Paragraph(child, doc)
        elif child.tag.endswith('}tbl'):
            yield Table(child, doc)

def extract_docx_blocks(doc: Document, max_text_len=2000):
    """遍历一次，顺序混排，文本自动合并成大块"""
    blocks = []
    elements = list(iter_block_items(doc))
    n = len(elements)
    para_buffer = []
    def flush_buffer():
        nonlocal para_buffer
        if para_buffer:
            content = "\n".join(para_buffer)
            while len(content) > max_text_len:
                blocks.append({"type": "text", "content": content[:max_text_len]})
                content = content[max_text_len:]
            if content.strip():
                blocks.append({"type": "text", "content": content})
            para_buffer = []

    for idx, item in enumerate(elements):
        if isinstance(item, Table):
            flush_buffer()
            table_name = ""
            for preidx in range(idx - 1, -1, -1):
                pre = elements[preidx]
                if isinstance(pre, Paragraph):
                    pre_text = pre.text.strip()
                    if pre_text:
                        table_name = pre_text
                        break
            df = [[cell.text.strip() for cell in row.cells] for row in item.rows]
            tbl_markdown = ""
            try:
                tbl_markdown = pd.DataFrame(df[1:], columns=df[0]).to_markdown(index=False) if len(df) > 1 else ""
            except Exception:
                pass
            blocks.append({
                "type": "table",
                "name": table_name,
                "markdown": tbl_markdown,
                "raw_df": df
            })
        elif isinstance(item, Paragraph):
            has_img = False
            for run in item.runs:
                if run.element.drawing_lst:
                    has_img = True
                    flush_buffer()
                    img_name = ""
                    for nxtidx in range(idx + 1, n):
                        nxt = elements[nxtidx]
                        if isinstance(nxt, Paragraph):
                            txt = nxt.text.strip()
                            if txt:
                                img_name = txt
                                break
                    for drawing in run.element.drawing_lst:
                        blip = drawing.xpath('.//a:blip')
                        if blip:
                            rId = blip[0].get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
                            image_part = doc.part.related_parts[rId]
                            img_bytes = image_part.blob
                            blocks.append({
                                "type": "image",
                                "name": img_name,
                                "img_bytes": img_bytes
                            })
            if not has_img and item.text.strip():
                para_buffer.append(item.text.strip())
    flush_buffer()
    return blocks

def split_text(text: str) -> list[str]:
    if not text:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,       # 块
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "；", "！", "？", " ", ""]
    )
    sub_chunks = splitter.create_documents([text])
    return [d.page_content.strip() for d in sub_chunks if d.page_content.strip()]

def find_optimal_clusters(embeddings, max_k=10):
    if len(embeddings) <= 1:
        return 1
    best_k = 1
    best_score = -1
    for k in range(2, min(max_k + 1, len(embeddings))):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(embeddings)
            labels = kmeans.labels_
            if len(set(labels)) > 1:
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
        except Exception as e:
            print(f"聚类异常 (k={k}): {e}")
            continue
    return best_k if best_score > 0 else 1

def semantic_grouping(embeddings, texts, optimal_k=None):
    if embeddings.size == 0 or not embeddings.any():
        return {}
    if optimal_k is None:
        optimal_k = find_optimal_clusters(embeddings)
    try:
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto').fit(embeddings)
        labels = kmeans.labels_
        groups = defaultdict(list)
        for i, label in enumerate(labels):
            groups[label].append({"text": texts[i], "embedding": embeddings[i], "index": i})
        return groups
    except Exception as e:
        print(f"语义分组失败: {e}")
        return {0: [{"text": texts[i], "embedding": embeddings[i], "index": i} for i in range(len(texts))]}



@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload(
    file: UploadFile = File(...),
    kb_id: str = Form(...),
    doc_type: Literal["text", "image", "table", "docx"] = Form("word"),
    key_words: str = Form(""),
    owner: str = Form("晓佳"),
):
    if not es.indices.exists(index=kb_id):
        raise HTTPException(404, f"知识库索引 {kb_id} 不存在")

    doc_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    filename = os.path.basename(file.filename)
    kw_list = [kw.strip() for kw in re.split(r"[，,]", key_words or "") if kw.strip()]
    actions = []

    if doc_type == "docx":
        file_bytes = await file.read()
        doc = Document(BytesIO(file_bytes))
        blocks = extract_docx_blocks(doc, max_text_len=2000)
        for idx, block in enumerate(blocks):
            # chunk_id = f"{doc_id}-{idx:04d}"
            block_order = idx  # 统一顺序  # 新增
            if block["type"] == "text":
                chunks = split_text(block["content"])
                embeddings, valid_chunks = [], []
                for chunk in chunks:
                    vec = embed(chunk, expected_dim=4096)
                    if vec:
                        embeddings.append(vec)
                        valid_chunks.append(chunk)
                for i, (chunk, vec) in enumerate(zip(valid_chunks, embeddings)):
                    split_index = i  # 新增
                    chunk_idx = f"{doc_id}-{block_order:04d}-{split_index:02d}"  # 新增
                    # chunk_idx = f"{chunk_id}-{i:02d}"
                    actions.append({
                        "_index": kb_id,
                        "_id": chunk_idx,
                        "_source": {
                            "kb_id": kb_id,
                            "doc_id": doc_id,
                            "chunk_id": chunk_idx,
                            "type": "text",
                            "title": filename,
                            "chunk_index": idx,
                            "content": chunk,
                            "keywords": kw_list,
                            "vector_text": vec,
                            "date": now,
                            "owner": owner,
                            "hits": 0
                        }
                    })
            elif block["type"] == "table":
                split_index = 0  # 新增
                chunk_id = f"{doc_id}-{block_order:04d}-{split_index:02d}"  # 新增
                table_for_emb = f"{block['name']}\n{block['markdown']}" if block.get("name") else block["markdown"]
                emb = embed(table_for_emb, expected_dim=4096) or [0.0] * 4096
                actions.append({
                    "_index": kb_id,
                    "_id": chunk_id,
                    "_source": {
                        "kb_id": kb_id,
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "type": "table",
                        "title": filename,
                        "chunk_index": idx,
                        "tbl_markdown": block["markdown"],
                        "tbl_name": block["name"],
                        "vector_table": emb,
                        "keywords": kw_list,
                        "date": now,
                        "owner": owner,
                        "hits": 0
                    }
                })
            elif block["type"] == "image":
                split_index = 0  # 新增
                chunk_id = f"{doc_id}-{block_order:04d}-{split_index:02d}"  # 新增

                fname = f"{uuid.uuid4().hex}.jpg"
                fpath = os.path.join(STATIC_IMG_DIR, fname)
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                with open(fpath, "wb") as f:
                    f.write(block["img_bytes"])
                image_url = f"{IMG_BASE_URL}/{fname}"
                content = block["name"] or ""
                print(f"Content: {content}")

                caption = generate_caption_from_vllm(block["img_bytes"])
                print(f"Caption: {caption}")

                img_title_for_emb = f"{content} {caption}" if content and caption else content or caption
                img_vec = embed(img_title_for_emb, expected_dim=4096) if img_title_for_emb else None
                actions.append({
                    "_index": kb_id,
                    "_id": chunk_id,
                    "_source": {
                        "kb_id": kb_id,
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "type": "image",
                        "title": filename,
                        "chunk_index": idx,
                        "image_path": image_url,
                        "content": content,           # 图名/图注，作为 content
                        "img_caption": caption,       # 生成的图片描述
                        "vector_image": img_vec,
                        "keywords": kw_list,
                        "date": now,
                        "owner": owner,
                        "hits": 0
                    }
                })
    elif doc_type == "text":
        raw = (await file.read()).decode("utf-8", "ignore")
        filtered = "\n".join([line.strip() for line in raw.splitlines() if line.strip()])
        chunks = split_text(filtered)
        embeddings, valid_chunks = [], []
        for chunk in chunks:
            vec = embed(chunk, expected_dim=4096)
            if vec:
                embeddings.append(vec)
                valid_chunks.append(chunk)
        if not embeddings:
            raise HTTPException(400, "所有分块嵌入失败")
        groups = semantic_grouping(np.array(embeddings), valid_chunks)
        for group_id, items in groups.items():
            for item in sorted(items, key=lambda x: x["index"]):
                chunk_id = f"{doc_id}-{item['index']:04d}"
                actions.append({
                    "_index": kb_id,
                    "_id": chunk_id,
                    "_source": {
                        "kb_id": kb_id,
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "type": "text",
                        "title": filename,
                        "chunk_index": item["index"],   # 保持和分块顺序一致
                        "content": item["text"],
                        "keywords": kw_list,
                        "vector_text": item["embedding"],
                        "date": now,
                        "owner": owner,
                        "hits": 0,
                    }
                })

    elif doc_type == "image":
        content = await file.read()
        ext = os.path.splitext(filename)[-1]
        fname = f"{uuid.uuid4().hex}{ext}"
        img_path = f"/img1/{fname}"
        full_path = f"/home/okok/nginx-static/static{img_path}"
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "wb") as f:
            f.write(content)

        caption = generate_caption_from_vllm(content)
        embedding = embed(caption, expected_dim=4096) if caption else [0.0] * 4096

        actions.append({
            "_index": kb_id,
            "_id": doc_id,
            "_source": {
                "kb_id": kb_id,
                "doc_id": doc_id,
                "chunk_id": doc_id,
                "type": "image",
                "title": filename,
                "chunk_index": 0,
                "image_path": f"http://10.199.194.246:8081{img_path}",
                "img_caption": caption,
                "img_ocr": "",
                "vector_image": embedding,
                "keywords": kw_list,
                "date": now,
                "owner": owner,
                "hits": 0
            }
        })

    elif doc_type == "table":
        file_bytes = await file.read()
        filename = file.filename.lower()
        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(BytesIO(file_bytes))
            elif filename.endswith(".xls") or filename.endswith(".xlsx"):
                df = pd.read_excel(BytesIO(file_bytes))
            else:
                raise HTTPException(400, "不支持的表格格式（仅支持 .csv, .xls, .xlsx）")
        except Exception as e:
            raise HTTPException(400, f"解析表格失败: {e}")
        markdown = df.to_markdown(index=False)
        headers = df.columns.tolist()
        header_str = ",".join([str(h) for h in headers])
        embedding = embed(markdown, expected_dim=4096) or [0.0] * 4096
        actions.append({
            "_index": kb_id,
            "_id": doc_id,
            "_source": {
                "kb_id": kb_id,
                "doc_id": doc_id,
                "chunk_id": doc_id,
                "type": "table",
                "title": filename,
                "chunk_index": 0,
                "tbl_markdown": markdown,
                "tbl_header": header_str,
                "vector_table": embedding,
                "keywords": kw_list,
                "date": now,
                "owner": owner,
                "hits": 0
            }
        })

    try:
        helpers.bulk(es, actions, request_timeout=120)
    except BulkIndexError as e:
        raise HTTPException(500, f"Bulk 写入失败: {e.errors}")

    return {
        "msg": "upload success",
        "kb_id": kb_id,
        "doc_id": doc_id,
        "filename": filename,
        "type": doc_type,
        "chunk_count": len(actions),
        "owner": owner,
    }
