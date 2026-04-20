# cli_search.py
import sys
from knowledge_base import KnowledgeBaseService
import config_data as cfg

def main():
    # 1. 初始化知识库（这里需要传入之前讨论过的持久化路径）
    kb = KnowledgeBaseService()

    print("=== 纯检索系统已启动 (输入 'exit' 退出) ===")
    
    while True:
        query = input("\n请输入搜索问题 > ")
        if query.lower() in ['exit', 'quit']:
            break
        
        if not query.strip():
            continue

        # 2. 执行检索
        print(f"正在检索与 '{query}' 最相关的片段...")
        results = kb.search_documents(query, top_k=3)

        # 3. 格式化输出结果
        if not results:
            print("未找到相关内容。")
        else:
            print(f"\n找到 {len(results)} 个匹配结果：")
            for i, doc in enumerate(results):
                source = doc.metadata.get("source", "未知来源")
                score = doc.metadata.get("score", "N/A") # 部分配置下可获取得分
                print(f"--- 结果 {i+1} [来源: {source}] ---")
                print(doc.page_content.strip())
                print("-" * 30)

if __name__ == "__main__":
    main()