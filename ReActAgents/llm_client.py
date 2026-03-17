import os
from openai import OpenAI, APIStatusError, APIConnectionError
from dotenv import load_dotenv
from typing import Any

# 加载 .env 文件中的环境变量
load_dotenv()

class HelloAgentsLLM:
    """
    为 "Hello Agents" 定制的LLM客户端。
    它用于调用任何兼容OpenAI接口的服务，并默认使用流式响应。
    """

    def __init__(self, model: str = None, api_key: str = None, base_url: str = None, timeout: int = None):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        api_key = api_key or os.getenv("LLM_API_KEY")
        base_url = base_url or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        # 检查必要的参数是否已提供
        if not all([self.model, api_key, base_url]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def think(self, messages: list[dict[str, str]], temperature: float = 0, stream: bool = True) -> str:
        """
        调用大语言模型进行思考，并返回其响应。
        """
        print(f"🧠 正在调用 {self.model} 模型进行思考...")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=stream,
            )
        
            # 处理流式响应
            print(f"✅ {self.model} 模型响应成功：")
            collected_chunks = []
            for chunk in response:
                chunks = chunk.choices[0].delta.content or ""
                print(chunks, end="", flush=True)
                collected_chunks.append(chunks)
            print()  # 在流式输出后换行

            return "".join(collected_chunks)

        except APIStatusError as e:
            # 处理 API 状态错误（如 400, 401, 403, 429, 500 等）
            status_code = e.status_code
            try:
                # 尝试获取 LLM 服务返回的具体错误代码 (JSON body 中的 code)
                error_details = e.response.json()
                llm_code = error_details.get("code", "N/A")
                llm_message = error_details.get("message", e.message)
                print(f"❌ API 错误 (HTTP {status_code}):")
                print(f"   - LLM 服务错误码: {llm_code}")
                print(f"   - 错误信息: {llm_message}")
            except Exception:
                print(f"❌ API 错误 (HTTP {status_code}): {e.message}")
            return None

        except APIConnectionError as e:
            # 处理网络连接问题
            print(f"❌ 网络连接错误：无法连接到 LLM 服务器。")
            print(f"   - 详情: {e}")
            return None

        except Exception as e:
            # 处理其他未知错误
            print(f"❌ 调用 {self.model} 模型时发生未知错误：{e}")
            return None


# --- 客户端使用示例 ---
if __name__ == '__main__':
    try:
        llm_client = HelloAgentsLLM()
        
        SYSTEM_PROMPT = "You are a helpful assistant that writes Python code."
        USER_PROMPT = input("User: ") or "Hello, how are you?"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]

        print("--- 调用 LLM ---")
        response_text = llm_client.think(messages, temperature=0.7)
        if response_text:
            print("\n\n--- LLM 完整响应 ---")
            print(response_text)

    except ValueError as e:
        print(e)
