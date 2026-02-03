# NLPAndAI
Purpose to study the machine learning and LNP related libs and tools

目前Python版本基于 3.11.9
具体安装教程参考以下：
https://ncnmfdan85y5.feishu.cn/wiki/HCfmwbdHziqIgikJ1C5celMWnHc

---

## VSCode 创建虚拟环境 (venv)

基于 `requirements.txt` 生成 venv 的几种方法：

### 方法一：VSCode 自动检测（推荐）

1. 打开项目文件夹后，VSCode 检测到 `requirements.txt` 会提示创建虚拟环境
2. 点击提示中的 **"Create Environment"**
3. 选择 **"Venv"**
4. 选择 Python 解释器版本
5. VSCode 会自动创建 venv 并安装依赖

### 方法二：命令行手动创建

```bash
# 1. 在项目根目录打开终端 (Ctrl+`)

# 2. 创建虚拟环境
python -m venv .venv

# 3. 激活虚拟环境（Windows）
.venv\Scripts\activate

# 或（macOS/Linux）
source .venv/bin/activate

# 4. 安装依赖
pip install -r requirements.txt
```

### 方法三：VSCode 命令面板

1. `Ctrl+Shift+P` 打开命令面板
2. 输入并选择：**"Python: Create Environment"**
3. 选择 **"Venv"**
4. 选择 Python 解释器
5. 勾选 **"requirements.txt"** 自动安装依赖

### 验证环境

```bash
# 查看当前使用的 Python 路径
which python

# 查看已安装的包
pip list
```

### VSCode 设置（可选）

确保 VSCode 使用该虚拟环境：

1. `Ctrl+Shift+P` → **"Python: Select Interpreter"**
2. 选择 `./.venv/Scripts/python.exe` (Windows) 或 `./.venv/bin/python` (Mac/Linux)

或在工作区设置 `.vscode/settings.json`：

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe"
}
```


