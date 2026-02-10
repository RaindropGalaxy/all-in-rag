import os
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

Settings.llm = OpenAILike(
    model="deepseek-chat",  # DeepSeek官方模型名
    api_key=os.getenv("DEEPSEEK_API_KEY"),  # 确保该密钥是DeepSeek官方有效密钥
    api_base="https://api.deepseek.com/v1",  # DeepSeek官方Base URL
    is_chat_model=True
)
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

documents = SimpleDirectoryReader(input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]).load_data()

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

print(query_engine.get_prompts())  # 打印默认提示模板
print("\n===================== 最终回答 =====================")
print(query_engine.query("文中举了哪些例子?"))  # 执行RAG查询并输出回答