# knowledge_base.py
import os
import hashlib
from datetime import datetime
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config_data as config

def get_string_md5(input_str: str) -> str:
    return hashlib.md5(input_str.encode('utf-8')).hexdigest()

def check_md5(md5_str: str):
    if not os.path.exists(config.md5_path):
        return False
    with open(config.md5_path, 'r', encoding='utf-8') as f:
        return md5_str in [line.strip() for line in f.readlines()]

def save_md5(md5_str: str):
    with open(config.md5_path, 'a', encoding='utf-8') as f:
        f.write(md5_str + '\n')

class KnowledgeBaseService:
    def __init__(self):
        os.makedirs(config.persist_directory, exist_ok=True)
        self.chroma = Chroma(
            collection_name=config.collection_name,
            embedding_function=DashScopeEmbeddings(model=config.embedding_model_name),
            persist_directory=config.persist_directory,
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators
        )

    def upload_by_str(self, data, file_name):
        md5_hex = get_string_md5(data)
        if check_md5(md5_hex):
            return "[跳过] 该文档内容已存在"

        chunks = self.spliter.split_text(data)
        metadata = {
            "source": file_name,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.chroma.add_texts(chunks, metadatas=[metadata for _ in chunks])
        save_md5(md5_hex)
        return f"[成功] 文档 {file_name} 已向量化存储，切分为 {len(chunks)} 块"