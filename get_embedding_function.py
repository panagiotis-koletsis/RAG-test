# from langchain_community.embeddings.bedrock import BedrockEmbeddings
#
#
# def get_embedding_function():
#     embeddings = BedrockEmbeddings(
#         credentials_profile_name="default", region_name="us-east-1"
#     )
from langchain_community.embeddings.ollama import OllamaEmbeddings

# embeddings = OllamaEmbeddings(model="llama3:8b")
# t=embeddings.embed_query("test")
# len(t)

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="mistral")
    return embeddings