from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530", db_name="rag_tutorial")
rows = client.query(
      collection_name="docs",
      filter="",
      output_fields=["id", "text", "source", "chunk_id"],
      limit=5,
)
print(rows)
client.close()