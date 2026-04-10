from qdrant_client import QdrantClient, models
import os

client = QdrantClient(url=os.getenv("https://30934615-32a7-49e8-b39f-ca7e8a4ce660.europe-west3-0.gcp.cloud.qdrant.io"), api_key=os.getenv("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6ZjI5MDU2M2MtNGRmMC00ZWE0LTg2NTItMzk3YTk3Mzg5NzYzIn0.HR9WA_mgrn_R_qdD5SH-0c1TlucIlwBqAcHriR84_Uk"))


# For Colab:
# from google.colab import userdata
# client = QdrantClient(url=userdata.get("QDRANT_URL"), api_key=userdata.get("QDRANT_API_KEY"))

# Quick health check
collections = client.get_collections()
print(f"Connected to Qdrant Cloud: {len(collections.collections)} collections")