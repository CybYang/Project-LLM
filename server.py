from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from pathlib import Path
import hashlib, json, os

app = Flask(__name__)
CORS(app)

# === 配置路径 ===
doc_folder = "documents"
faiss_index_path = "faiss_index"
hash_log_path = "processed_hashes.json"

# === 嵌入模型与切分器 ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)

# === 哈希工具 ===
def file_hash(path: Path):
    return hashlib.md5(path.read_bytes()).hexdigest()

def load_hash_log():
    return json.loads(Path(hash_log_path).read_text()) if Path(hash_log_path).exists() else {}

def save_hash_log(log):
    Path(hash_log_path).write_text(json.dumps(log, indent=2), encoding="utf-8")

def load_new_documents():
    docs, prev, curr = [], load_hash_log(), {}
    for file in Path(doc_folder).glob("**/*"):
        if file.suffix.lower() not in [".txt", ".md", ".pdf"]:
            continue
        try:
            fid = file_hash(file)
            curr[str(file)] = fid
            if prev.get(str(file)) == fid:
                continue
            loader = TextLoader(str(file), encoding="utf-8") if file.suffix != ".pdf" else PyPDFLoader(str(file))
            for d in loader.load():
                d.metadata["source"] = str(file)
                docs.append(d)
        except Exception as e:
            print(f"❌ 加载失败：{file.name}：{e}")
    save_hash_log(curr)
    return docs

# === 初始化向量库 ===
if Path(faiss_index_path).exists():
    print("📦 加载已有向量库...")
    vs = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)
    new_docs = load_new_documents()
    if new_docs:
        vs.add_documents(text_splitter.split_documents(new_docs))
        vs.save_local(faiss_index_path)
else:
    print("🧱 构建新向量库...")
    docs = text_splitter.split_documents(load_new_documents())
    vs = FAISS.from_documents(docs, embedding_model)
    vs.save_local(faiss_index_path)

llm = Ollama(model="deepseek-r1:7b")
retriever = vs.as_retriever()

# === 保留运行时记忆（仅内存、3轮）
chat_history = []

# server.py 修改后的 chat 接口部分（其余内容不变）
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("question", "").strip()
    if not query:
        return jsonify({"answer": "❗问题不能为空"}), 400

    retrieved = retriever.get_relevant_documents(query)
    doc_context = "\n".join([d.page_content.strip() for d in retrieved[:3]])

    context_prompt = ""
    for item in chat_history[-3:]:
        context_prompt += f"用户：{item['question']}\nAI：{item['answer']}\n"

    final_prompt = ""
    if doc_context:
        final_prompt += f"以下是一些可能有用的参考资料：\n{doc_context}\n\n"
    final_prompt += context_prompt + f"用户：{query}\nAI："

    full_answer = llm.invoke(final_prompt)

    # 分离 <think> 标签内容
    import re
    match = re.search(r"<think>(.*?)</think>(.*)", full_answer, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        answer = match.group(2).strip()
    else:
        thinking = ""
        answer = full_answer.strip()

    chat_history.append({"question": query, "answer": full_answer})
    return jsonify({
        "thinking": thinking,
        "answer": answer
    })

@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")

if __name__ == "__main__":
    app.run(port=5000)
