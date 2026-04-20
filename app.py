# app.py
import os
import streamlit as st
from knowledge_base import KnowledgeBaseService
from rag import RagService
import config_data as cfg
import tempfile

st.set_page_config(page_title="DeepSeek RAG 系统", layout="wide")

# 初始化 Service
if "kb_service" not in st.session_state:
    st.session_state["kb_service"] = KnowledgeBaseService()
if "rag_service" not in st.session_state:
    st.session_state["rag_service"] = RagService()

# app.py
with st.sidebar:
    st.header("知识库管理")
    uploaded_file = st.file_uploader("上传资料", type=["txt", "pdf"]) # 增加 pdf 类型
    
    if uploaded_file:
        if st.button("更新至向量库"):
            with st.spinner("正在处理文档..."):
                if uploaded_file.type == "application/pdf":
                    # PDF 需要通过文件路径读取，创建临时文件
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    msg = st.session_state["kb_service"].upload_pdf(tmp_path, uploaded_file.name)
                    os.remove(tmp_path) # 清理临时文件
                else:
                    # 原有的 TXT 处理逻辑
                    content = uploaded_file.getvalue().decode("utf-8")
                    msg = st.session_state["kb_service"].upload_by_str(content, uploaded_file.name)
                
                st.success(msg)

st.title("🤖 DeepSeek 智能 RAG 问答")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "您好！知识库已就绪，请提问。"}]

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("请输入您的问题"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # 使用流式输出增加交互感
        for chunk in st.session_state["rag_service"].chain.stream(
            {"input": prompt}, config=cfg.session_config
        ):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
        st.session_state["messages"].append({"role": "assistant", "content": full_response})