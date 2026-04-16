# app.py
import streamlit as st
from knowledge_base import KnowledgeBaseService
from rag import RagService
import config_data as cfg

st.set_page_config(page_title="DeepSeek RAG 系统", layout="wide")

# 初始化 Service
if "kb_service" not in st.session_state:
    st.session_state["kb_service"] = KnowledgeBaseService()
if "rag_service" not in st.session_state:
    st.session_state["rag_service"] = RagService()

# 侧边栏：文件上传 [cite: 70]
with st.sidebar:
    st.header("知识库管理")
    uploaded_file = st.file_uploader("上传 TXT 资料", type=["txt"])
    if uploaded_file:
        content = uploaded_file.getvalue().decode("utf-8")
        if st.button("更新至向量库"):
            with st.spinner("处理中..."):
                msg = st.session_state["kb_service"].upload_by_str(content, uploaded_file.name)
                st.success(msg)

# 主界面：对话 [cite: 63]
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
        
        # 使用流式输出增加交互感 [cite: 67]
        for chunk in st.session_state["rag_service"].chain.stream(
            {"input": prompt}, config=cfg.session_config
        ):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
        st.session_state["messages"].append({"role": "assistant", "content": full_response})