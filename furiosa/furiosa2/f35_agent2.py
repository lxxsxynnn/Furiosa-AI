from common import get_llm
from langchain_core.tools import tool
from langchain.agents import create_agent
# pip install langchain

@tool
def add(a:int, b:int) -> int:
    """두 정수를 더합니다."""
    return a + b

@tool
def multiply(a:int, b:int) -> int:
    """두 정수를 곱합니다."""
    return a * b

llm = get_llm(max_tokens=500)

agent = create_agent(
    model=llm,
    tools=[add, multiply],
    system_prompt="계산이 필요하면 도구를 사용하세요."
    # 에이전트한테 역할과 행동 지침을 설정
)

result = agent.invoke({
    "messages": [
        {"role":"user", "content":"33과 7을 곱한 후 4를 더해줘."}
    ]
})

# print(result)
print("========================")
from langchain_core.messages import AIMessage
for i, msg in enumerate(result["messages"]):
    print(f"{i+1}번째 메시지")
    print("type: ", type(msg).__name__)
    # print("content: ", msg.content)

    if isinstance(msg, AIMessage) and msg.tool_calls:
        for call in msg.tool_calls:
            print("tool name: ", call['name'])
            print("parameters: ", call['args'])
        

'''
========================
1번째 메시지
type:  HumanMessage
2번째 메시지
type:  AIMessage
tool name:  multiply
parameters:  {'a': 33, 'b': 7}
tool name:  add
parameters:  {'a': 231, 'b': 4}
3번째 메시지
type:  ToolMessage
4번째 메시지
type:  ToolMessage
5번째 메시지
type:  AIMessage
========================
'''

print("========================")
# print("final response: ", result["messages"][-1].content)
'''
final response:  <think>
Okay, let me go through this step by step. The user asked to multiply 33 by 7 and then add 4. First, I need to handle the multiplication. The tool provided for multiplication is 'multiply' which takes two integers. So I'll call multiply(33, 7). The result of that is 231. Then, the next step is to add 4 to that result. The 'add' tool is available for this, so I use add(231, 4). The sum is 235. The user probably wants the final answer, so I'll present that clearly.
</think>

33과 7을 곱하면 231이 되고, 그 결과에 4를 더하면 **235**입니다.
'''