import os
from openai import OpenAI

from dotenv import load_dotenv
# 把 .env 里所有 KEY=VALUE 注入到 os.environ
load_dotenv()          # 默认查找当前目录下的 .env
#print("已加载API密钥：", os.getenv("DASHSCOPE_API_KEY"))

print("正在初始化OpenAI客户端...")

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
)

completion = client.embeddings.create(
    model="text-embedding-v4",
    input='我想知道迪士尼的退票政策',
    dimensions=1024, # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
    encoding_format="float"
)

print(completion.model_dump_json())