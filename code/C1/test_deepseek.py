from openai import OpenAI
import sys
import io

# 1. 彻底重定向标准输出，强制使用 UTF-8 编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')

# 2. 配置 DeepSeek 客户端（替换成你的真实 Key）
client = OpenAI(
    api_key="sk-fa31788ff2c04e6d82916d9b28d94f87",  # 你的完整 Key（带 sk- 前缀）
    base_url="https://api.deepseek.com/v1"
)

# 3. 测试调用（捕获原始错误信息）
try:
    # 发送简单请求（用英文提问，避免中文编码提前触发报错）
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Hello"}]  # 先用英文测试
    )
    # 输出结果（强制转码）
    result = response.choices[0].message.content
    print(f"✅ 调用成功！回复：{result}")
    
except Exception as e:
    # 输出原始错误（不转中文，先看真实认证问题）
    error_msg = str(e).encode('utf-8', errors='ignore').decode('utf-8')
    print(f"❌ 调用失败，原始错误：{error_msg}")