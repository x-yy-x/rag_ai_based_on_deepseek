# config_data.py
md5_path = "./md5.text"

# Chroma 配置
collection_name = "deepseek_rag"
persist_directory = "./chroma_db"

# 文本切分配置
chunk_size = 500 
chunk_overlap = 50
separators = ["\n\n", "\n", "。", "！", "？", " ", ""]
max_split_char_number = 500

# 检索配置
similarity_threshold = 3  # 对应 top-k=3

# 模型配置
# 嵌入模型建议使用本地 bge 或兼容 API
embedding_model_name = "text-embedding-v4" 
# 确保环境变量已配置 DEEPSEEK_API_KEY
chat_model_name = "deepseek-chat" 

session_config = {
    "configurable": {
        "session_id": "user_session_001",
    }
}