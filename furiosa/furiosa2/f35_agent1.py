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

print("========================")
# print(result)
print("========================")

for i, msg in enumerate(result["messages"]):
    print(f"{i+1}번째 메시지")
    print("type: ", type(msg).__name__)
    print("content: ", msg.content)

'''
1번째 메시지
type:  HumanMessage
content:  33과 7을 곱한 후 4를 더해줘.
2번째 메시지
type:  AIMessage
content:  <think>
Okay, let's see. The user wants me to multiply 33 by 7 and then add 4. Hmm, first I need to handle the multiplication. The tools provided have a 'multiply' function that takes two integers. So I should call multiply with 33 and 7. Then, once I get the result of that, I need to add 4. The 'add' function is available for that. So first step: multiply 33 and 7. Let me check the parameters again to make sure. Both functions require two integers, which I have. So the first tool call is multiply(33, 7). The result of that would be 231. Then, add 4 to 231, which would be 235. So the final answer should be 235. Let me make sure I'm not missing any steps here. The user's query is straightforward: multiply 33 and 7, then add 4. So two function calls in sequence. Alright, I'll start by calling the multiply function.
</think>


3번째 메시지
type:  ToolMessage
content:  231
4번째 메시지
type:  ToolMessage
content:  235
5번째 메시지
type:  AIMessage
content:  <think>
Okay, let me go through this step by step. The user asked to multiply 33 by 7 and then add 4. First, I need to handle the multiplication. The tool provided for multiplication is 'multiply' which takes two integers. So I'll call multiply(33, 7). The result of that is 231. Then, the next step is to add 4 to that result. The 'add' tool is available for this, so I use add(231, 4). The sum is 235. The user probably wants the final answer, so I'll present that clearly.
</think>

33과 7을 곱하면 231이 되고, 그 결과에 4를 더하면 **235**입니다.
'''

print("========================")
print("final response: ", result["messages"][-1].content)
'''
final response:  <think>
Okay, let me go through this step by step. The user asked to multiply 33 by 7 and then add 4. First, I need to handle the multiplication. The tool provided for multiplication is 'multiply' which takes two integers. So I'll call multiply(33, 7). The result of that is 231. Then, the next step is to add 4 to that result. The 'add' tool is available for this, so I use add(231, 4). The sum is 235. The user probably wants the final answer, so I'll present that clearly.
</think>

33과 7을 곱하면 231이 되고, 그 결과에 4를 더하면 **235**입니다.
'''