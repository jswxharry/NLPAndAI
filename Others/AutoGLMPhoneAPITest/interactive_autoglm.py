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
        self.vm_ready = False  # VM 初始化完成标志
        self.msg_counter = 0
        self.lock = threading.Lock()
        self.print_lock = threading.Lock()  # 专门用于同步打印输出
        self.current_task = None  # 当前执行的任务
        self.task_finished = True  # 任务是否完成
        self.last_action = None  # 上一次操作类型，用于合并相同操作
        self.action_count = 0  # 相同操作计数
        self.waiting_input = False  # 是否正在等待用户输入
        
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
                # 结果消息 - 提取关键信息以可读格式显示
                self._display_result(data)
            else:
                # 其他消息 - 简化显示
                self._display_simple_message(data, msg_type)
        except json.JSONDecodeError:
            print(f"\n[📩 原始消息] {message}")
    
    def _safe_print(self, *args, **kwargs):
        """线程安全的打印"""
        with self.print_lock:
            print(*args, **kwargs)
    
    def _display_result(self, data: dict):
        """以可读格式显示执行结果"""
        result_data = data.get('data', {})
        result_type = result_data.get('result_type', 'unknown')
        
        self._safe_print(f"\n{'='*60}")
        self._safe_print(f"[✅ 执行结果] 类型: {result_type}")
        self._safe_print(f"{'='*60}")
        
        # 根据结果类型提取关键信息
        if result_type == 'text':
            content = result_data.get('content', '')
            self._safe_print(f"📄 内容:\n{content}")
        elif result_type == 'image':
            image_url = result_data.get('url', '')
            self._safe_print(f"🖼️  图片地址: {image_url}")
        elif result_type == 'error':
            error_msg = result_data.get('error', '未知错误')
            self._safe_print(f"❌ 错误: {error_msg}")
        else:
            # 通用处理 - 显示 data 中的所有字段
            for key, value in result_data.items():
                if key == 'result_type':
                    continue
                if isinstance(value, str) and len(value) > 200:
                    self._safe_print(f"📋 {key}:\n{value[:200]}...")
                else:
                    self._safe_print(f"📋 {key}: {value}")
        
        # 显示消息元信息
        self._safe_print(f"{'='*60}")
        self._safe_print(f"🕐 时间戳: {data.get('timestamp')}")
        self._safe_print(f"🆔 消息ID: {data.get('msg_id', 'N/A')}")
        self._safe_print(f"{'='*60}")
        self._safe_print(f"\n💡 提示: 输入指令继续，或输入 'quit' 退出")
    
    def _display_simple_message(self, data: dict, msg_type: str):
        """简化显示其他消息"""
        msg_data = data.get('data', {})
        
        # 检查是否是初始化相关消息，如果是则简化显示
        if msg_type in ('server_init', 'server_session'):
            self._display_init_message(msg_type, msg_data)
            return
        
        # 客户端发送确认消息 - 服务端回执，不重复显示
        if msg_type == 'client_test':
            # 指令发送确认已在 send_instruction 时显示，这里只重置任务状态
            with self.lock:
                self.task_finished = False
            return
        
        # 服务端任务消息
        if msg_type == 'server_task':
            self._display_task_message(msg_data)
            return
        
        # server_notify 消息简化显示
        if msg_type == 'server_notify':
            self._display_notify_message(msg_data)
            return
        
        # 其他消息简化显示
        self._safe_print(f"\n[📩 {msg_type}]")
        msg_id = data.get('msg_id', 'N/A')
        timestamp = data.get('timestamp')
        
        if msg_data:
            if isinstance(msg_data, dict):
                for key, value in msg_data.items():
                    if isinstance(value, (dict, list)):
                        summary = self._summarize_value(value)
                        self._safe_print(f"  • {key}: {summary}")
                    else:
                        self._safe_print(f"  • {key}: {value}")
            else:
                self._safe_print(f"  • 数据: {msg_data}")
        
        self._safe_print(f"  • 消息ID: {msg_id}")
        if timestamp:
            self._safe_print(f"  • 时间戳: {timestamp}")
    
    def _display_init_message(self, msg_type: str, msg_data: dict):
        """显示初始化消息（简洁格式）"""
        biz_type = msg_data.get('biz_type', '')
        vm_state = msg_data.get('vm_state', '')
        
        # 根据消息类型显示进度
        if msg_type == 'server_init':
            self._safe_print(f"  🔄 服务初始化中...")
        elif msg_type == 'server_session':
            if biz_type == 'init_vm':
                self._safe_print(f"  🔄 正在启动虚拟机...")
            elif biz_type == 'init_session':
                if vm_state == 'vm_successful':
                    with self.lock:
                        if not self.vm_ready:
                            self.vm_ready = True
                            self._safe_print(f"  ✅ 虚拟机就绪")
                else:
                    self._safe_print(f"  🔄 虚拟机状态: {vm_state}")
    
    def _display_task_message(self, msg_data: dict):
        """显示任务执行消息"""
        biz_type = msg_data.get('biz_type', '')
        data_agent_str = msg_data.get('data_agent', '{}')
        
        try:
            data_agent = json.loads(data_agent_str) if isinstance(data_agent_str, str) else data_agent_str
        except json.JSONDecodeError:
            data_agent = {}
        
        action = data_agent.get('action', '')
        
        # 合并连续相同的操作
        if action and action == self.last_action and action in ('tap', 'click', 'wait'):
            self.action_count += 1
            # 使用回车符覆盖上一行显示计数
            print(f"\r  🤖 执行操作: {action} (x{self.action_count})", end='', flush=True)
            return
        else:
            # 如果之前有合并的操作，先换行
            if self.action_count > 1:
                print()  # 换行结束合并显示
            self.last_action = action
            self.action_count = 1
        
        # 根据 action 类型显示不同状态
        if action == 'home':
            self._safe_print(f"\n  📱 执行操作: 返回桌面")
            self.last_action = action
        elif action == 'finish':
            with self.lock:
                self.task_finished = True
            # 如果有合并的操作，先换行
            if self.action_count > 1:
                self._safe_print()
            self._safe_print(f"\n  ✅ 任务执行完毕")
            self.last_action = None
            self.action_count = 0
        elif action == 'tap' or action == 'click':
            x, y = data_agent.get('x', 0), data_agent.get('y', 0)
            if x and y:
                self._safe_print(f"\r  👆 点击: ({x}, {y})      ", end='', flush=True)
            else:
                self._safe_print(f"\r  🤖 执行操作: tap", end='', flush=True)
        elif action == 'input' or action == 'type':
            text = data_agent.get('text', '')
            self._safe_print(f"\n  ⌨️  输入: {text[:30]}{'...' if len(text) > 30 else ''}")
            self.last_action = action
        elif action == 'swipe':
            direction = data_agent.get('direction', '')
            start_x = data_agent.get('start_x', 0)
            start_y = data_agent.get('start_y', 0)
            end_x = data_agent.get('end_x', 0)
            end_y = data_agent.get('end_y', 0)
            
            # 根据坐标计算方向
            if direction and direction != 'unknown':
                self._safe_print(f"\n  👋 滑动: {direction}")
            elif start_x and end_x:
                if end_x > start_x + 100:
                    self._safe_print(f"\n  👋 向右滑动")
                elif start_x > end_x + 100:
                    self._safe_print(f"\n  👋 向左滑动")
                elif end_y > start_y + 100:
                    self._safe_print(f"\n  👋 向下滑动")
                elif start_y > end_y + 100:
                    self._safe_print(f"\n  👋 向上滑动")
                else:
                    self._safe_print(f"\n  👋 滑动: ({start_x},{start_y}) -> ({end_x},{end_y})")
            else:
                self._safe_print(f"\n  👋 滑动操作")
            self.last_action = action
        elif action == 'long_press':
            x, y = data_agent.get('x', 0), data_agent.get('y', 0)
            self._safe_print(f"\n  👇 长按: ({x}, {y})")
            self.last_action = action
        elif action == 'launch':
            app = data_agent.get('app', '')
            self._safe_print(f"\n  🚀 启动应用: {app}")
            self.last_action = action
        elif action == 'wait':
            self._safe_print(f"\r  ⏳ 等待...", end='', flush=True)
        elif action:
            self._safe_print(f"\n  🤖 执行操作: {action}")
            self.last_action = action
        else:
            # 无 action，显示简略信息
            if biz_type:
                self._safe_print(f"\n  📋 任务类型: {biz_type}")
    
    def _display_notify_message(self, msg_data: dict):
        """显示通知消息（简洁格式）"""
        biz_type = msg_data.get('biz_type', '')
        query_status = msg_data.get('query_status', '')
        reason = msg_data.get('reason', '')
        
        # 只显示关键状态变化
        if biz_type == 'notify_task':
            if query_status == 'task_doing':
                # 任务进行中，只在特定情况下显示
                pass  # 不显示，避免刷屏
            elif query_status == 'task_done' or reason == 'finished':
                self._safe_print(f"\n  📋 任务状态: 已完成")
    
    def _summarize_value(self, value, max_len: int = 100) -> str:
        """对复杂值生成简短摘要"""
        text = json.dumps(value, ensure_ascii=False)
        if len(text) <= max_len:
            return text
        return text[:max_len] + f"... (共 {len(text)} 字符)"
    
    def on_open(self, ws):
        """连接打开时的回调"""
        self.connected = True
        self._safe_print("✅ WebSocket 连接已建立")
        self._safe_print("⏳ 正在初始化服务，请稍候...")
        
    def on_error(self, ws, error):
        """发生错误时的回调"""
        self._safe_print(f"\n❌ 错误: {error}")
        
    def on_close(self, ws, close_status_code, close_msg):
        """连接关闭时的回调"""
        self.connected = False
        self._safe_print(f"\n🔌 连接已关闭")
        if close_status_code:
            self._safe_print(f"   状态码: {close_status_code}")
        if close_msg:
            self._safe_print(f"   原因: {close_msg}")
    
    def send_instruction(self, instruction: str):
        """发送指令"""
        if not self.connected or not self.ws:
            self._safe_print("❌ 未连接到服务器，无法发送指令")
            return False
            
        msg = self.create_message(instruction)
        # 重置任务状态（必须在发送前重置）
        with self.lock:
            self.task_finished = False
            self.last_action = None
            self.action_count = 0
        
        # 显示发送信息
        self._safe_print(f"\n[📤 发送指令 #{self.msg_counter}] {instruction}")
        self._safe_print("-" * 60)
        self._safe_print(f"✅ 指令已发送: {instruction[:40]}{'...' if len(instruction) > 40 else ''}")
        self._safe_print("⏳ 等待任务执行...")
        
        self.ws.send(json.dumps(msg))
        return True
    
    def show_help(self):
        """显示帮助信息"""
        self._safe_print("\n" + "="*60)
        self._safe_print("📖 AutoGLM 交互式客户端 - 帮助")
        self._safe_print("="*60)
        self._safe_print("可用命令:")
        self._safe_print("  <任意文本>   - 发送指令给 AutoGLM")
        self._safe_print("  help        - 显示此帮助信息")
        self._safe_print("  status      - 查看连接状态")
        self._safe_print("  example     - 显示示例指令")
        self._safe_print("  quit/exit   - 退出程序")
        self._safe_print("="*60)
    
    def show_examples(self):
        """显示示例指令"""
        examples = [
            "帮我在小红书找三篇云南的旅游攻略汇总一篇",
            "打开微信，给张三发消息说晚上一起吃饭",
            "在美团搜索附近的火锅店",
            "打开淘宝搜索 iPhone 16 的价格",
            "在抖音搜索美食视频",
        ]
        self._safe_print("\n" + "="*60)
        self._safe_print("📝 示例指令:")
        self._safe_print("="*60)
        for i, ex in enumerate(examples, 1):
            self._safe_print(f"  {i}. {ex}")
        self._safe_print("="*60)
    
    def run(self):
        """运行交互式客户端"""
        self._safe_print("\n" + "="*60)
        self._safe_print("🚀 AutoGLM Phone API 交互式客户端")
        self._safe_print("="*60)
        self._safe_print(f"连接地址: {URL}")
        self._safe_print(f"API Key: {API_KEY[:10]}...{API_KEY[-4:]}")
        self._safe_print("-" * 60)
        
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
            self._safe_print("❌ 连接超时")
            return
        
        # 等待 VM 初始化完成
        init_timeout = 60
        start_time = time.time()
        while not self.vm_ready and time.time() - start_time < init_timeout:
            time.sleep(0.1)
        
        if not self.vm_ready:
            self._safe_print("❌ 服务初始化超时")
            return
        
        # 初始化完成，显示提示信息
        self._safe_print("-" * 60)
        self._safe_print("💡 提示: 输入指令发送给 AutoGLM，输入 'quit' 或 'exit' 退出")
        self._safe_print("💡 提示: 输入 'help' 查看帮助")
        self._safe_print("-" * 60)
        
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
                        self._safe_print(f"\n连接状态: {status}")
                    elif user_input.lower() == 'example':
                        self.show_examples()
                    else:
                        # 发送指令
                        if self.send_instruction(user_input):
                            # 等待任务执行完成
                            wait_timeout = 120  # 最长等待120秒
                            wait_start = time.time()
                            while not self.task_finished and time.time() - wait_start < wait_timeout:
                                time.sleep(0.1)
                            if not self.task_finished:
                                self._safe_print("\n⚠️  任务执行超时，但仍然可以发送下一条指令")
                            else:
                                # 任务完成，显示分隔线和提示
                                self._safe_print(f"\n{'-'*60}")
                                self._safe_print(f"💡 提示: 输入下一条指令，或输入 'quit' 退出")
                                self._safe_print(f"{'-'*60}")
                        
                except EOFError:
                    # 处理 Ctrl+D
                    break
                    
        except KeyboardInterrupt:
            self._safe_print("\n\n👋 用户中断")
        finally:
            self.ws.close()
            self._safe_print("👋 已断开连接，再见！")


def main():
    """主函数"""
    client = AutoGLMInteractiveClient()
    client.run()


if __name__ == "__main__":
    main()
