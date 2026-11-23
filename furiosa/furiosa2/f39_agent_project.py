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
    system_prompt=(
        "당신은 계산 전문 Agent입니다. "
        "계산이 필요하면 등록된 도구를 사용하세요. "
        "여러 단계의 계산은 순서대로 수행하세요. "
        "계산 오류가 발생하면 사용자에게 설명하세요."
    )
    # 에이전트한테 역할과 행동 지침을 설정
)

# Agent 함수
def run_agent(question):
    print("user question: ", question)
    print("========================")

    result = agent.invoke(
        {
            "messages": [
                {"role":"user", "content":question}
            ]
        },
        config={"recursion_limit": 20}
    )


    print("========================")
    for i, msg in enumerate(result["messages"]):
        print(f"{i+1}번째 메시지")
        print("type: ", type(msg).__name__)

        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                print("tool name: ", call['name'])
                print("parameters: ", call['args'])

    print("========================")
    print("final response: ", result["messages"][-1].content)
    return result

# 사용자 입력
while True:
    question = input("\n계산할 질문을 입력하세요(종료: q): ")

    if question.lower() =='q':
        print("Agent 종료")
        break

    if not question.strip():
        continue

    try:
        run_agent(question)
    except Exception as e:
        print("오류 발생: ", type(e).__name__, e)