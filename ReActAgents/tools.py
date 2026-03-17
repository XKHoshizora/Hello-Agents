import os
from dotenv import load_dotenv
from serpapi import SerpApiClient
from typing import Any

# 加载 .env 文件中的环境变量
load_dotenv()

def search(query: str) -> str:
    """
    一个基于SerpApi的实战网页搜索引擎工具。
    它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    """
    print(f"🔍 正在执行 [SerpApi] 网页检索：{query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "❌ 错误：未找到 “SerpAPI API Key，请检查 .env 文件。"

        params = {
            "q": query,
            "engine": "google",
            "api_key": api_key,
            "hl": "zh-CN",  # 语言代码
            "gl": "cn",  # 国家代码
        }
        
        client = SerpApiClient(params)
        results = client.get_dict()

        # 智能解析：优先寻找最直接的答案或知识图谱信息
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # 如果没有答案，则返回前三个有机结果的摘要
            snippet = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}\n{res.get('link', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]

            return "\n\n".join(snippet)

        return f"⚠️ 未找到与 “{query}” 相关的信息。"
        
    except Exception as e:
        return f"❌ 执行网页检索时出错：{str(e)}"


class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具。
    """
    def __init__(self):
        self.tools: dict[str, dict[str, Any]] = {}

    def register_tool(self, name: str, description: str, func: callable) -> None:
        """
        向工具箱中注册一个新工具。
        """
        if name in self.tools:
            if input(f"工具 {name} 已存在，是否跳过注册：（y/n）") != "n":
                print(f"工具 {name} 已跳过注册。")
                return
            else:
                print(f"工具 {name} 将被覆盖。")
        
        self.tools[name] = {"description": description, "func": func}
        print(f"🎉 工具 {name} 已成功注册。")

    def get_tool(self, name: str) -> callable:
        """
        根据名称获取一个工具的执行函数。
        """
        return self.tools.get(name, {}).get("func")

    def get_available_tools(self) -> str:
        """
        获取所有可用工具的格式化描述字符串。
        """
        return "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])


# --- 工具初始化与使用示例 ---
if __name__ == '__main__':
    # 1. 初始化工具执行器
    tool_executor = ToolExecutor()

    # 2. 注册我们的搜索工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.register_tool("Search", search_description, search)
    
    # 3. 打印可用的工具
    print("\n--- 可用的工具 ---")
    print(tool_executor.get_available_tools())

    # 4. 智能体的Action调用，这次我们问一个实时性的问题
    print("\n--- 执行 Action: Search['英伟达最新的GPU型号是什么'] ---")
    tool_name = "Search"
    tool_input = "英伟达最新的GPU型号是什么"

    tool_function = tool_executor.get_tool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("--- 观察 (Observation) ---")
        print(observation)
    else:
        print(f"错误：未找到名为 '{tool_name}' 的工具。")
