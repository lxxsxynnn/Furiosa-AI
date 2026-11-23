from common import get_llm
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import AIMessage

@tool
def add(a:int, b:int) -> int:
    """두 정수를 더합니다."""
    return a + b

@tool
def subtract(a:int, b:int) -> int:
    """첫 번째 정수에서 두 번째 정수를 뺍니다."""
    return a - b

@tool
def multiply(a:int, b:int) -> int:
    """두 정수를 곱합니다."""
    return a * b

@tool
def divide(a:int, b:int) -> float:
    """첫 번째 정수를 두 번째 정수로 나눕니다."""
    return a / b

llm = get_llm(max_tokens=500)

tools=[add, subtract, multiply, divide]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="계산이 필요하면 도구를 사용하세요."
    # 에이전트한테 역할과 행동 지침을 설정
)

questions = [
    "100에서 35를 빼줘.",
    "12와 8을 곱해줘.",
    "100을 4로 나눠줘.",
    "15와 27을 더해줘.",
]

for question in questions:
    print("\nQuestion: ", question)

    result = agent.invoke({
        "messages": [
            {"role":"user", "content":question}
        ]
    })

    # print(result)
    print("========================")
    for i, msg in enumerate(result["messages"]):
        print(f"{i+1}번째 메시지")
        print("type: ", type(msg).__name__)
        # print("content: ", msg.content)

        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                print("tool name: ", call['name'])
                print("parameters: ", call['args'])

    print("========================")
    print("final response: ", result["messages"][-1].content)
    print("========================")

'''
Question:  100에서 35를 빼줘.
========================
1번째 메시지
type:  HumanMessage
2번째 메시지
type:  AIMessage
tool name:  subtract
parameters:  {'a': 100, 'b': 35}
3번째 메시지
type:  ToolMessage
4번째 메시지
type:  AIMessage
========================
final response:  <think>
Okay, the user asked to subtract 35 from 100. I used the subtract function with a=100 and b=35. The tool response was 65. So the answer is 65. I should just present that clearly.
</think>

100에서 35를 뺀 결과는 **65**입니다.
========================

Question:  12와 8을 곱해줘.
========================
1번째 메시지
type:  HumanMessage
2번째 메시지
type:  AIMessage
tool name:  multiply
parameters:  {'a': 12, 'b': 8}
3번째 메시지
type:  ToolMessage
4번째 메시지
type:  AIMessage
========================
final response:  <think>
Okay, the user asked to multiply 12 and 8. I called the multiply function with a=12 and b=8. The tool response was 96. So the answer is 96. I should just present that clearly.
</think>

12와 8의 곱은 **96**입니다.
========================

Question:  100을 4로 나눠줘.
========================
1번째 메시지
type:  HumanMessage
2번째 메시지
type:  AIMessage
tool name:  divide
parameters:  {'a': 100, 'b': 4}
3번째 메시지
type:  ToolMessage
4번째 메시지
type:  AIMessage
========================
final response:  <think>
Okay, the user asked to divide 100 by 4. I used the divide function with a=100 and b=4. The result from the tool was 25.0. Now I need to present this answer clearly. Let me check if there's any formatting needed. The user just wants the result, so I'll state it directly. 100 divided by 4 is 25.0. That's straightforward.
</think>

100을 4로 나누면 25.0입니다.
========================

Question:  15와 27을 더해줘.
========================
1번째 메시지
type:  HumanMessage
2번째 메시지
type:  AIMessage
tool name:  add
parameters:  {'a': 15, 'b': 27}
3번째 메시지
type:  ToolMessage
4번째 메시지
type:  AIMessage
========================
final response:  <think>
Okay, the user asked to add 15 and 27. I called the 'add' function with a=15 and b=27. The tool response was 42. So the sum is 42. I should just present that as the answer. No need for extra steps here. Just confirm the result of the addition.
</think>

15와 27을 더하면 42입니다.
========================
'''
