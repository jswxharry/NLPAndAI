"""
AutoGLM Phone API WebSocket 测试客户端
通过 WebSocket 连接 AutoGLM API 并发送测试指令
"""

import os
import json
import time
import uuid
import websocket
from dotenv import load_dotenv

from dotenv import load_dotenv
# 把 .env 里所有 KEY=VALUE 注入到 os.environ
load_dotenv()          # 默认查找当前目录下的 .env

# 从环境变量获取 API Key
API_KEY = os.getenv("AUTO_GLM_API_KEY")
if not API_KEY:
    raise ValueError("未找到 API Key，请在 .env 文件中设置 AUTO_GLM_API_KEY")

# WebSocket URL
URL = "wss://autoglm-api.zhipuai.cn/openapi/v1/autoglm/developer"

# 请求头
HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}


def create_test_message(instruction: str = "帮我在小红书找三篇云南的旅游攻略汇总一篇") -> dict:
    """
    创建测试指令消息
    
    Args:
        instruction: 要执行的任务指令
        
    Returns:
        消息字典
    """
    return {
        "timestamp": int(time.time() * 1000),
        "conversation_id": "",
        "msg_type": "client_test",
        "msg_id": str(uuid.uuid4()),
        "data": {
            "biz_type": "test_agent",
            "instruction": instruction
        }
    }


def on_message(ws, message):
    """收到消息时的回调"""
    try:
        # 尝试解析 JSON
        data = json.loads(message)
        print(f"\n[收到消息] {json.dumps(data, ensure_ascii=False, indent=2)}")
    except json.JSONDecodeError:
        print(f"\n[收到消息] {message}")


def on_open(ws):
    """连接打开时的回调"""
    print("✓ WebSocket 连接已建立")
    
    # 发送测试指令
    test_msg = create_test_message()
    print(f"\n[发送指令] {json.dumps(test_msg, ensure_ascii=False, indent=2)}")
    ws.send(json.dumps(test_msg))


def on_error(ws, error):
    """发生错误时的回调"""
    print(f"✗ 错误: {error}")


def on_close(ws, close_status_code, close_msg):
    """连接关闭时的回调"""
    print(f"\n✗ 连接已关闭")
    if close_status_code:
        print(f"  状态码: {close_status_code}")
    if close_msg:
        print(f"  原因: {close_msg}")


def run_test(instruction: str = None):
    """
    运行 WebSocket 测试
    
    Args:
        instruction: 自定义指令，不指定则使用默认指令
    """
    print(f"=" * 50)
    print("AutoGLM Phone API WebSocket 测试")
    print(f"=" * 50)
    print(f"连接地址: {URL}")
    print(f"API Key: {API_KEY[:10]}...{API_KEY[-4:]}")
    print(f"-" * 50)
    
    # 创建 WebSocket 应用
    ws = websocket.WebSocketApp(
        URL,
        header=HEADERS,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    
    try:
        # 运行 WebSocket 连接
        ws.run_forever()
    except KeyboardInterrupt:
        print("\n\n用户中断，正在关闭连接...")
        ws.close()


if __name__ == "__main__":
    import sys
    
    # 支持命令行参数传入自定义指令
    custom_instruction = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    
    run_test(custom_instruction)
