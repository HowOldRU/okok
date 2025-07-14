import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.kb     import router as kb_router
from routes.upload import router as upload_router
from routes.doc    import router as doc_router
from routes.search import router as search_router

app = FastAPI(title="KB API", version="0.4.0")

# 配置适用于开发环境，允许跨域请求，且没有任何限制
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)

app.include_router(kb_router)
app.include_router(upload_router)
app.include_router(doc_router)
app.include_router(search_router)



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8500, reload=True)
