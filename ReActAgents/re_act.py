import re
from llm_client import HelloAgentsLLM
from tools import ToolExecutor

# ReAct Prompt Template
REACT_PROMPT_TEMPLATE="""
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下：
{tools}

请严格按照以下格式进行回应：

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一：
- `{{tool_name}}[{{tool_input}}]`：调用一个可用工具。
- `Finish[最终答案]`：当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在`Action:`字段后使用 `finish(answer="...")` 来输出你总结出的最终答案。


现在，请开始解决以下问题：
Question: {question}
History: {history}
"""

system_prompt = REACT_PROMPT_TEMPLATE

class ReActAgent:
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5, system_prompt: str = system_prompt):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []
        self.system_prompt = system_prompt

    def run(self, question: str) -> str:
        """
        执行 ReAct 智能体来回答问题。
        """
        self.history = [] # 重置历史记录
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第 {current_step} 步 ---")

            # 1.格式化提示词
            tools_description = self.tool_executor.get_available_tools()
            history_str = "\n".join(self.history)
            prompt = self.system_prompt.format(
                tools=tools_description,
                history=history_str,
                question=question
            )

            # 2. 调用 LLM 进行思考
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.think(messages=messages)

            if not response:
                print(f"❌ 第 {current_step} 步失败：LLM 未返回响应。")
                break

            # 3. 解析 LLM 的输出
            thought, action = self._parse_output(response)

            if thought:
                print(f"🤔 思考：{thought}")

            if not action:
                print(f"⚠️ 警告：未能解析出有效的行动（Action:{action}），智能体将停止思考。")
                break

            # 4. 执行行动（Action）
            if action.startswith("Finish"):
                # 如果是 Finish 行动，提取最终答案并结束智能体运行
                final_answer = re.match(r"Finish\[(.*)\]", action).group(1)
                print(f"🎉 最终答案：\n{final_answer}")
                return final_answer

            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                # 处理无效的行动（Action）格式
                print(f"⚠️ 警告：未能解析出有效的行动（Action:{action}），智能体将跳过当前步骤。")
                continue

            print(f"🚀 执行行动：{tool_name}[{tool_input}]")

            # 5. 工具调用
            tool_function = self.tool_executor.get_tool(tool_name)  
            if not tool_function:
                print(f"⚠️ 警告：未能找到工具（Tool:{tool_name}）。")
                observation = f"⚠️ 警告：未能找到工具（Tool:{tool_name}）。"
            else:
                # 执行工具调用
                observation = tool_function(tool_input)

            # 6. 观察结果
            print(f"👀 观察：\n{observation}")

            # 7. 更新历史记录，为下一次思考做准备
            # 将本轮的行动（Action）和观察（Observation）添加到历史记录中
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")
        
        # 循环结束
        print("--- 智能体执行结束 ---")
        return "⚠️ 智能体思考次数超过限制，流程终止。"

    def _parse_output(self, text: str) -> tuple[str, str]:
        """
        解析 LLM 的输出，提取智能体的思考(Thought)和行动(Action)。
        """
        # 使用正则表达式匹配思考(Thought)和行动(Action)
        thought_match = re.search(r"Thought:\s*(.*)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(.*)", text, re.DOTALL)
        
        # 提取思考(Thought)和行动(Action)
        thought = thought_match.group(1).strip() if thought_match else ""
        action = action_match.group(1).strip() if action_match else ""
        
        return thought, action
        
    def _parse_action(self, action_text: str) -> tuple[str, str]:
        """
        解析行动(Action)的文本，提取工具名称和输入参数。
        """
        match = re.match(r"(\w+)\[(.*)\]", action_text)

        if match:
            return match.group(1), match.group(2)
        
        return "", ""


# --- ReAct 智能体使用示例 ---
if __name__ == '__main__':
    from tools import search

    try:
        # 初始化 LLM 客户端
        llm_client = HelloAgentsLLM(model="THUDM/GLM-4-9B-0414")

        # 初始化工具执行器，并注册工具
        tool_executor = ToolExecutor()
        search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
        tool_executor.register_tool("Search", search_description, search)
        
        # 初始化 ReAct 智能体
        re_act_agent = ReActAgent(llm_client=llm_client, tool_executor=tool_executor)

        # 获取用户输入
        question = input("User: ") or "Apple 最新的手机是哪一款？它的主要卖点是什么？你不清楚现在实际是几几年时，搜索不要加年份。"

        # 运行智能体
        answer = re_act_agent.run(question)
        print("\n\n--- 🎉 最终答案 ---")
        print(answer)

    except ValueError as e:
        print(e)
