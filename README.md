## 🤖 本地知识库 AI 助手

这是一个基于 Flask + LangChain + FAISS + Ollama 的本地 AI 问答系统，支持加载本地文档进行向量检索，并通过本地大模型进行回答。配套网页前端，界面美观，支持代码高亮与数学公式展示。

------

### 🔧 项目特点

- 本地文档上传与增量更新（支持 `.txt`, `.md`, `.pdf`）
- 基于 FAISS 构建向量数据库，支持快速语义检索
- 使用 Ollama 本地模型 `deepseek-r1:7b` 进行 AI 回答
- 保留最近三轮上下文，实现短期记忆
- 网页端支持气泡式聊天 UI、Markdown 渲染、代码高亮
- 结构清晰，便于拓展与二次开发

------

### 📂 项目结构

```
├── documents/                 // 用于存放自己想用来训练的文档
│   ├── smaple.txt
│   ├── sample.md
│   └── sample.pdf
│   └── ...
├── fass_index/                // 向量库（运行后产生）
│   ├── index.faiss
│   ├── index.pkl
├── chat_memory.json           // 记录问答记录（运行后产生）
├── processed_hashes.json      // 记录处理过的文档（运行后产生）
├── run.py                     // 命令行版
├── server.py                  // 与 index.html 共同构成网页版
├── index.html
└── README.md
```

------

### 🚀 使用方法

####  ✅  安装依赖

访问 [ollama](https://ollama.com/) 官网下载软件

运行软件，拉取模型 `ollama run deepseek-r1:7b`  

此时 ollama 已经可以运行并且可以在命令行中与模型对话但无法使用 RAG 训练

若想使用 RAG 训练应退出 ollama 

在命令行中运行 `ollama serve` 以启动 ollama 的后台监听服务。

接下来安装依赖 `pip install flask flask-cors langchain faiss-cpu sentence-transformers` 

实际需要根据自己的 python 环境进行补充

####  ▶  启动服务

- **命令行**：
  - 命令行中输入 `python run.py` 
- **网页**：
  - 命令行中输入 `python server.py` 
  - 用浏览器访问 `http://localhost:5000/` 
- **数据加载**：
  - 程序启动可以正常与模型对话，并且提问到相关问题时，命令行版与网页版会自动共用 `documents` 下的文件作为 RAG 进行训练。



### 🧠 交互说明

- 用户输入问题后，系统会自动提取相关文档片段并结合上下文生成回答。
- 若回答中包含 `思考内容`，则“AI思考过程”会以灰色文字显示。
- 支持自动渲染 Markdown 。



### 📌 技术栈

| 类别     | 技术框架![example](example.png)                              |
| -------- | ------------------------------------------------------------ |
| 后端服务 | Flask + Flask-CORS                                           |
| AI模型   | Ollama + DeepSeek-R1                                         |
| 文档处理 | LangChain + FAISS + HuggingFace Embedding                    |
| 前端页面 | 原生 HTML + CSS + JavaScript                                 |
| 渲染支持 | Marked.js（Markdown）、highlight.js（代码高亮）、MathJax（公式渲染） |

------

### 📸 示例截图

![](F:\Haoyang_Sun\programming_task\20250605-llm(ds)\example2.png)

![](F:\Haoyang_Sun\programming_task\20250605-llm(ds)\example1.png)

