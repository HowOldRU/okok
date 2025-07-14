# import requests
# import time  # 导入time模块用于计时
#
# EMBEDDING_URL = "http://10.146.138.192:11434/api/embeddings"
# EMBEDDING_MODEL = "qwen3:8b"
#
#
# def get_embedding(text: str):
#     start_time = time.time()  # 记录开始时间
#
#     resp = requests.post(
#         EMBEDDING_URL,
#         json={"model": EMBEDDING_MODEL, "prompt": text},
#         timeout=10
#     )
#     resp.raise_for_status()
#     emb = resp.json()["embedding"]
#
#     end_time = time.time()  # 记录结束时间
#     elapsed_time = end_time - start_time  # 计算耗时
#
#     print(f"Embedding length: {len(emb)}")
#     print("Embedding preview:", emb[:10])
#     print(f"Embedding耗时: {elapsed_time:.3f}秒")  # 显示耗时，保留3位小数
#
#     return emb
#
#
# if __name__ == "__main__":
#     get_embedding("测试一下bge-m3模型的embedding维度是多少")






#  写入图片
# from elasticsearch import Elasticsearch
#
# es = Elasticsearch("http://10.199.194.246:9200")
#
# es.index(
#     index="your_index_name",
#     document={
#         "title": "朱晓佳的QQ头像",
#         "content": "朱晓佳的QQ头像",
#         "image_url": "http://10.199.194.246:8081/img/b_a20446b5dd01368a537de186525de7fe.jpg"
#     }
# )
#




# import base64
# import requests
#
# # ====== 配置 ======
# VLLM_URL = "http://10.199.194.246:3002/v1/chat/completions"  # OpenAI 兼容路径
# EMBEDDING_URL = "http://10.146.138.192:11434/api/embeddings"
# EMBEDDING_MODEL = "qwen3:8b"
# IMAGE_FILE = "/home/okok/nginx-static/static/img/b_a20446b5dd01368a537de186525de7fe.jpg"
# EXPECTED_DIM = 768
#
# # ====== Step 1: 读取并编码图片 ======
# with open(IMAGE_FILE, "rb") as f:
#     img_bytes = f.read()
#     base64_image = base64.b64encode(img_bytes).decode()
#     image_data_url = f"data:image/jpeg;base64,{base64_image}"
#
# # ====== Step 2: 请求 vLLM 生成 caption ======
# print("[1] 调用 vLLM 生成图像 caption...")
# payload = {
#     "model": "/models/Qwen2.5-VL-7B-Instruct",
#     "messages": [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image_url", "image_url": {"url": image_data_url}},
#                 {"type": "text", "text": "请描述这张图片"}
#             ]
#         }
#     ],
#     "temperature": 0.7,
#     "max_tokens": 128
# }
#
# try:
#     resp = requests.post(VLLM_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
#     resp.raise_for_status()
#     caption = resp.json()["choices"][0]["message"]["content"].strip()
#     print("Caption:", caption)
# except Exception as e:
#     print("❌ caption 生成失败：", e)
#     exit(1)
#
# # ====== Step 3: 请求 embedding ======
# print("[2] 调用嵌入服务...")
# embed_payload = {
#     "model": EMBEDDING_MODEL,
#     "prompt": caption
# }
#
# try:
#     resp = requests.post(EMBEDDING_URL, json=embed_payload, timeout=60)
#     resp.raise_for_status()
#     embedding = resp.json().get("embedding", [])
#     print("向量维度:", len(embedding))
#     if len(embedding) != EXPECTED_DIM:
#         print(f"❌ 向量维度异常！期望 {EXPECTED_DIM}，实际 {len(embedding)}")
#     else:
#         print("✅ 向量维度符合预期")
# except Exception as e:
#     print("❌ 嵌入失败：", e)


from docx import Document
from pathlib import Path
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.text.paragraph import Paragraph
from docx.table import Table


def extract_docx_to_txt(docx_path: str, output_txt_path: str):
    doc = Document(docx_path)
    lines = []

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if text:
                lines.append(text)

        elif isinstance(block, Table):
            md_table = []
            for row in block.rows:
                cells = [cell.text.strip().replace('\n', ' ') for cell in row.cells]
                md_table.append('| ' + ' | '.join(cells) + ' |')

            # 添加 Markdown 表头分隔符（如果表格至少两行）
            if len(md_table) >= 2:
                num_cols = md_table[0].count('|') - 1
                header_sep = '| ' + ' | '.join(['---'] * num_cols) + ' |'
                md_table.insert(1, header_sep)

            lines.append("\n".join(md_table))

    # 写入 TXT 文件
    Path(output_txt_path).write_text("\n\n".join(lines), encoding='utf-8')
    print(f"提取完成：{output_txt_path}")


def iter_block_items(doc):
    """
    按照文档顺序（段落+表格）逐个迭代
    """
    for child in doc.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield Table(child, doc)


# 示例用法
if __name__ == "__main__":
    extract_docx_to_txt("example.docx", "output.txt")
