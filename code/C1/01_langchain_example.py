import os
import sys
# ===================== 新增：强制UTF-8编码（解决中文输出报错） =====================
# 修复Codespaces终端中文编码问题，必须放在最开头
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

# hugging face镜像设置，如果国内环境无法使用启用该设置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 建议启用，避免模型下载失败

from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# ===================== 修正：加载环境变量 + 兜底配置 =====================
# 加载.env文件（如果存在），同时设置兜底值避免Key为空
load_dotenv()
# 优先从环境变量取Key，没有则用你验证过的有效Key（临时兜底）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "[你的API]")

markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"

# ===================== 新增：文件存在性检查（避免路径错误） =====================
if not os.path.exists(markdown_path):
    print(f"❌ 错误：找不到markdown文件，路径：{markdown_path}")
    print("请检查文件路径是否正确，确保easy-rl-chapter1.md存在于指定目录")
    sys.exit(1)

# 加载本地markdown文件
try:
    loader = UnstructuredMarkdownLoader(markdown_path)
    docs = loader.load()
    if not docs:
        print("⚠️ 警告：markdown文件加载成功，但内容为空")
except Exception as e:
    print(f"❌ 加载markdown文件失败：{str(e)}")
    sys.exit(1)

# 文本分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,    # 新增：明确分块大小（默认值不清晰，中文建议500字符）
    chunk_overlap=50   # 新增：分块重叠（避免语义割裂）
)
chunks = text_splitter.split_documents(docs)
print(f"✅ 文本分块完成，共生成 {len(chunks)} 个文本块")

# 中文嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
  
# 构建向量存储
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)
print(f"✅ 向量库构建完成，共存入 {len(chunks)} 个文本块")

# 提示词模板（格式优化，避免换行问题）
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文，且使用简洁、清晰的中文表述。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:""")

# ===================== 修正：LLM配置（使用验证过的DeepSeek配置） =====================
# 注释掉AIHubmix（避免混淆），使用你验证成功的DeepSeek官方接口
# 使用 AIHubmix（注释掉，暂不用）
# llm = ChatOpenAI(
#     model="glm-4.7-flash-free",
#     temperature=0.7,
#     max_tokens=4096,
#     api_key=DEEPSEEK_API_KEY,
#     base_url="https://aihubmix.com/v1"
# )

# 使用 DeepSeek 官方接口（已验证有效）
llm = ChatOpenAI(
    model="deepseek-chat",          # 必须和DeepSeek官方模型名一致
    temperature=0.7,
    max_tokens=4096,
    api_key=DEEPSEEK_API_KEY,       # 使用兜底后的有效Key
    base_url="https://api.deepseek.com/v1",  # 官方base_url（已验证）
    timeout=30                      # 新增：超时设置（避免请求卡死）
)

# 用户查询
question = "文中举了哪些例子？"

# ===================== 新增：检索结果检查 =====================
# 在向量存储中查询相关文档
retrieved_docs = vectorstore.similarity_search(question, k=3)
if not retrieved_docs:
    print("⚠️ 警告：未检索到与问题相关的文档，回答将为空")
    docs_content = ""
else:
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    print(f"✅ 检索到 {len(retrieved_docs)} 条相关文档")

# ===================== 修正：调用LLM并捕获异常 =====================
try:
    # 格式化prompt并调用LLM
    final_prompt = prompt.format(question=question, context=docs_content)
    answer = llm.invoke(final_prompt)
    # 输出最终回答（提取content字段，避免打印整个对象）
    print("\n===================== 最终回答 =====================")
    print(answer.content)
except Exception as e:
    print(f"\n❌ 调用大模型失败：{str(e)}")
    print("请检查：1. API Key是否有效 2. 网络是否正常 3. DeepSeek额度是否充足")
