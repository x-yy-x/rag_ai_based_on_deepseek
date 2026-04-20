import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

def start_chat():
    messages = [
        {"role": "system", "content": "你是一个乐于助人的 AI 助手。"}
    ]
    
    print("--- 已进入 DeepSeek 对话模式 (输入 'quit' 或 'exit' 退出) ---")

    while True:
        user_input = input("\n用户: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("对话结束。")
            break
        
        if not user_input:
            continue

        # 将用户输入添加到历史记录中
        messages.append({"role": "user", "content": user_input})

        try:
            # 1. 开启 stream=True 
            response = client.chat.completions.create(
                model="deepseek-chat", 
                messages=messages,
                temperature=1.3,
                stream=True  # 开启流式传输
            )

            print("\nDeepSeek: ", end="", flush=True)
            full_reply = ""

            # 2. 迭代处理返回的每一个数据块
            for chunk in response:
                chunk_content = chunk.choices[0].delta.content
                if chunk_content:
                    print(chunk_content, end="", flush=True)
                    full_reply += chunk_content

            messages.append({"role": "assistant", "content": full_reply})

        except Exception as e:
            if "402" in str(e):
                print("\n[提示] 余额不足：请检查 DeepSeek 账户余额或更换 API Key。")
            else:
                print(f"\n发生错误: {e}")

if __name__ == "__main__":
    start_chat()