# rag.py
from langchain_deepseek import ChatDeepSeek # 需安装 pip install langchain-deepseek
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
import config_data as config
from file_history_store import get_history

class RagService:
    def __init__(self):
        # 初始化检索器
        vector_db = Chroma(
            collection_name=config.collection_name,
            embedding_function=DashScopeEmbeddings(model=config.embedding_model_name),
            persist_directory=config.persist_directory
        )
        self.retriever = vector_db.as_retriever(search_kwargs={"k": config.similarity_threshold})

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的 AI 助手。请严格根据以下【参考内容】回答问题。
            1. 如果参考内容中没有相关信息，请直接说“根据现有资料无法回答”，不要编造。
            2. 给出准确、简洁的回答。
            3. 必须说明信息来源（例如“根据文档：xxx...”）。
            
            【参考内容】：
            {context}"""),
            MessagesPlaceholder("history"),
            ("user", "{input}")
        ])

        # 初始化 DeepSeek 模型
        self.chat_model = ChatDeepSeek(model=config.chat_model_name, temperature=0.3)
        self.chain = self._get_chain()

    def _get_chain(self):
        def format_docs(docs):
            return "\n\n".join([f"内容：{d.page_content}\n来源：{d.metadata.get('source', '未知')}" for d in docs])

        # 构建基础链
        core_chain = (
            {
                "context": (lambda x: x["input"]) | self.retriever | format_docs,
                "input": lambda x: x["input"],
                "history": lambda x: x["history"]
            }
            | self.prompt_template
            | self.chat_model
            | StrOutputParser()
        )

        # 封装历史记录
        return RunnableWithMessageHistory(
            core_chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history"
        )