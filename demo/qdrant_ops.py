import os
from dotenv import load_dotenv, find_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# 1. 读取我们之前配置好的 .env 文件里的云端密钥
load_dotenv(find_dotenv())
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# 2. 连接到你的 Qdrant Cloud 集群
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)


# 3. 核心步骤：创建一个适配你模型的 Collection
COLLECTION_NAME = "thesis_rag_demo"
YOUR_MODEL_DIMENSION = 768  # 👈 这里填入你模型实际的输出维度

# 检查是否已经存在，不存在则创建
if not client.collection_exists(collection_name=COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=YOUR_MODEL_DIMENSION, 
            distance=Distance.COSINE  # 推荐使用余弦相似度
        )
    )
    print(f"成功创建 Collection: {COLLECTION_NAME}")
else:
    print(f"Collection {COLLECTION_NAME} 已存在。")