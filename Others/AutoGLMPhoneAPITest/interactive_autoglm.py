"""
AutoGLM Phone API 交互式命令行客户端
支持在命令行发送指令并实时查看消息调用结果
"""

import os
import json
import time
import uuid
import threading
import websocket
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

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


class AutoGLMInteractiveClient:
    """AutoGLM 交互式客户端"""
    
    def __init__(self):
        self.ws = None
        self.connected = False
        self.msg_counter = 0
        self.lock = threading.Lock()
        
    def create_message(self, instruction: str) -> dict:
        """创建指令消息"""
        self.msg_counter += 1
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
    
    def on_message(self, ws, message):
        """收到消息时的回调"""
        try:
            data = json.loads(message)
            msg_type = data.get('msg_type', 'unknown')
            
            # 根据消息类型显示不同的格式
            if msg_type == 'heartbeat':
                # 心跳消息简化显示
                with self.lock:
                    print(f"\n[💓 心跳] {data.get('timestamp')}")
            elif msg_type == 'result':
                # 结果消息高亮显示
                print(f"\n{'='*60}")
                print(f"[✅ 执行结果] 类型: {data.get('data', {}).get('result_type', 'unknown')}")
                print(f"{'='*60}")
                print(json.dumps(data, ensure_ascii=False, indent=2))
                print(f"{'='*60}")
                print(f"\n💡 提示: 输入指令继续，或输入 'quit' 退出")
            else:
                # 其他消息
                print(f"\n[📩 消息-{msg_type}]")
                print(json.dumps(data, ensure_ascii=False, indent=2))
        except json.JSONDecodeError:
            print(f"\n[📩 原始消息] {message}")
    
    def on_open(self, ws):
        """连接打开时的回调"""
        self.connected = True
        print("✅ WebSocket 连接已建立")
        print("-" * 60)
        print("💡 提示: 输入指令发送给 AutoGLM，输入 'quit' 或 'exit' 退出")
        print("💡 提示: 输入 'help' 查看帮助")
        print("-" * 60)
        
    def on_error(self, ws, error):
        """发生错误时的回调"""
        print(f"\n❌ 错误: {error}")
        
    def on_close(self, ws, close_status_code, close_msg):
        """连接关闭时的回调"""
        self.connected = False
        print(f"\n🔌 连接已关闭")
        if close_status_code:
            print(f"   状态码: {close_status_code}")
        if close_msg:
            print(f"   原因: {close_msg}")
    
    def send_instruction(self, instruction: str):
        """发送指令"""
        if not self.connected or not self.ws:
            print("❌ 未连接到服务器，无法发送指令")
            return False
            
        msg = self.create_message(instruction)
        print(f"\n[📤 发送指令 #{self.msg_counter}] {instruction}")
        print("-" * 60)
        self.ws.send(json.dumps(msg))
        return True
    
    def show_help(self):
        """显示帮助信息"""
        print("\n" + "="*60)
        print("📖 AutoGLM 交互式客户端 - 帮助")
        print("="*60)
        print("可用命令:")
        print("  <任意文本>   - 发送指令给 AutoGLM")
        print("  help        - 显示此帮助信息")
        print("  status      - 查看连接状态")
        print("  example     - 显示示例指令")
        print("  quit/exit   - 退出程序")
        print("="*60)
    
    def show_examples(self):
        """显示示例指令"""
        examples = [
            "帮我在小红书找三篇云南的旅游攻略汇总一篇",
            "打开微信，给张三发消息说晚上一起吃饭",
            "在美团搜索附近的火锅店",
            "打开淘宝搜索 iPhone 16 的价格",
            "在抖音搜索美食视频",
        ]
        print("\n" + "="*60)
        print("📝 示例指令:")
        print("="*60)
        for i, ex in enumerate(examples, 1):
            print(f"  {i}. {ex}")
        print("="*60)
    
    def run(self):
        """运行交互式客户端"""
        print("\n" + "="*60)
        print("🚀 AutoGLM Phone API 交互式客户端")
        print("="*60)
        print(f"连接地址: {URL}")
        print(f"API Key: {API_KEY[:10]}...{API_KEY[-4:]}")
        print("-" * 60)
        
        # 创建 WebSocket 应用
        self.ws = websocket.WebSocketApp(
            URL,
            header=HEADERS,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        # 在后台线程运行 WebSocket
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        
        # 等待连接建立
        timeout = 10
        start_time = time.time()
        while not self.connected and time.time() - start_time < timeout:
            time.sleep(0.1)
        
        if not self.connected:
            print("❌ 连接超时")
            return
        
        # 交互式循环
        try:
            while self.connected:
                try:
                    # 获取用户输入
                    user_input = input("\n🔹 请输入指令: ").strip()
                    
                    if not user_input:
                        continue
                    
                    # 处理特殊命令
                    if user_input.lower() in ('quit', 'exit', 'q'):
                        print("👋 正在退出...")
                        break
                    elif user_input.lower() == 'help':
                        self.show_help()
                    elif user_input.lower() == 'status':
                        status = "🟢 已连接" if self.connected else "🔴 未连接"
                        print(f"\n连接状态: {status}")
                    elif user_input.lower() == 'example':
                        self.show_examples()
                    else:
                        # 发送指令
                        self.send_instruction(user_input)
                        
                except EOFError:
                    # 处理 Ctrl+D
                    break
                    
        except KeyboardInterrupt:
            print("\n\n👋 用户中断")
        finally:
            self.ws.close()
            print("👋 已断开连接，再见！")


def main():
    """主函数"""
    client = AutoGLMInteractiveClient()
    client.run()


if __name__ == "__main__":
    main()
