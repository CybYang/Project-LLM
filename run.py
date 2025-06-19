from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from pathlib import Path
import hashlib
import json
import os

# === 路径设置 ===
doc_folder = "documents"
faiss_index_path = "faiss_index"
hash_log_path = "processed_hashes.json"

# === 嵌入模型和文本切分器 ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)

# === 哈希函数用于文件去重 ===
def file_hash(file_path: Path) -> str:
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()

# === 读取/保存 文档哈希记录 ===
def load_hash_log():
    if Path(hash_log_path).exists():
        return json.loads(Path(hash_log_path).read_text())
    return {}

def save_hash_log(hash_log):
    with open(hash_log_path, "w", encoding="utf-8") as f:
        json.dump(hash_log, f, indent=2)

# === 载入文档 ===
def load_new_or_changed_documents():
    all_docs = []
    previous_hashes = load_hash_log()
    current_hashes = {}
    
    for file_path in Path(doc_folder).glob("**/*"):
        suffix = file_path.suffix.lower()
        if suffix not in [".txt", ".md", ".pdf"]:
            continue

        try:
            file_id = file_hash(file_path)
            current_hashes[str(file_path)] = file_id

            if previous_hashes.get(str(file_path)) == file_id:
                continue  # 未变更，跳过

            if suffix in [".txt", ".md"]:
                loader = TextLoader(str(file_path), encoding="utf-8")
            else:
                loader = PyPDFLoader(str(file_path))

            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = str(file_path)
            all_docs.extend(docs)

        except Exception as e:
            print(f"❌ 加载失败：{file_path.name}，原因：{e}")

    save_hash_log(current_hashes)
    return all_docs

# === 载入或构建向量库 ===
if Path(faiss_index_path).exists():
    print("📦 加载已有向量库...")
    vector_store = FAISS.load_local(
        faiss_index_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    print("📄 检查是否有新增文档...")
    new_docs = load_new_or_changed_documents()
    if new_docs:
        split_docs = text_splitter.split_documents(new_docs)
        vector_store.add_documents(split_docs)
        vector_store.save_local(faiss_index_path)
else:
    print("🧱 正在构建新向量库...")
    all_docs = load_new_or_changed_documents()
    split_docs = text_splitter.split_documents(all_docs)
    vector_store = FAISS.from_documents(split_docs, embedding_model)
    vector_store.save_local(faiss_index_path)

# === 初始化模型和QA链 ===
llm = Ollama(model="deepseek-r1:7b")
retriever = vector_store.as_retriever()
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# === 当前运行中的记忆（只保留最近 3 轮） ===
chat_history = []

# === 启动交互 ===
print("🔍 请输入你的问题，输入 'exit' 退出，输入 'reset' 清空记忆：")
while True:
    query = input("你：").strip()
    if query.lower() in ["exit", "quit"]:
        print("👋 再见！")
        break
    if query.lower() == "reset":
        chat_history = []
        print("🧠 已清空记忆。")
        continue

    # 文档检索（仍然基于 query 本身）
    retrieved_docs = retriever.get_relevant_documents(query)
    # print("\n📎 命中参考片段：")
    # shown = False
    # for i, doc in enumerate(retrieved_docs):
    #     content = doc.page_content.strip()
    #     if not content:
    #         continue
    #     source = doc.metadata.get("source", "未知来源")
    #     print(f"【文本{i+1}】(来自：{Path(source).name})\n{content}\n")
    #     shown = True
    # if not shown:
    #     print("（未找到明显相关的片段）")

    # 拼接记忆上下文（仅本轮运行的最后 3 轮）
    context_prompt = ""
    for item in chat_history[-3:]:
        context_prompt += f"用户：{item['question']}\nAI：{item['answer']}\n"

    # 拼接文档内容
    doc_context = "\n".join([doc.page_content.strip() for doc in retrieved_docs[:3]])
    final_prompt = ""
    if doc_context:
        final_prompt += f"以下是一些可能有用的参考资料：\n{doc_context}\n\n"
    final_prompt += context_prompt + f"用户：{query}\nAI："

    # 获取回答
    print("\n🤖 AI 回答：")
    answer = llm.invoke(final_prompt)
    print(answer)

    # 只保留本轮运行中的对话历史（内存中）
    chat_history.append({"question": query, "answer": answer})
