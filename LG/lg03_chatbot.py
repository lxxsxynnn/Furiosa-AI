# LangGraph 기본 챗봇 : State → Node → Edge → compile → invoke 순서로 노드 하나짜리 그래프 만들기
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# monogpt api 키 연결
load_dotenv()
api_key = os.environ['MONOROUTER_API_KEY'].strip()  # strip() : 앞뒤 공백·줄바꿈 제거
base_url = 'https://monogpt.kr/api/monorouter/v1'

llm = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url
)

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

################ 1. State 정의 ################
class State(TypedDict):
    messages : Annotated[list, add_messages]    # 노드가 돌려준 메시지를 기존 리스트 뒤에 붙임
    aaa : str   # 리듀서가 없는 키는 노드가 돌려준 값으로 덮어씀
    bbb : str
    ccc : str

# 실행 중 State 값은 이런 dict (TypedDict는 키와 타입만 정함)
# state = {
#     "messages" : [],
#     'aaa' : 'abc',
#     'bbb' : 'abc',
#     'ccc' : 'abc',
# }

################ 2. Node 정의 ################
def chatbot(state: State):     # State를 받아서 바꿀 키만 dict로 돌려줌
    return {"messages" : [llm.invoke(state["messages"])]}

################ 3. Graph 구성 ################
graph_01 = StateGraph(State)

graph_01.add_node("chatbot", chatbot)   # (노드 이름, 실행할 함수)

################ 4. Edge 추가 ################
graph_01.add_edge(START, "chatbot")
graph_01.add_edge("chatbot", END)

################ 5. Graph 컴파일 ################
graph = graph_01.compile()

################ 6. Graph 실행 ################
query = "삼성전자의 주가를 예측해줘"

response = graph.invoke({
        "messages" : [("user", query)]  # ("user", 문자열) 튜플은 add_messages가 HumanMessage로 바꿈
})

print(response)     # aaa·bbb·ccc는 값을 넣은 적이 없어서 결과에 없음
'''
{'messages': [
    HumanMessage(content='삼성전자의 주가를 예측해줘', additional_kwargs={}, response_metadata={}, id='69ebb2fd-...'),
    AIMessage(content='삼성전자 주가는 정확히 “예측”하기보다 **메모리 반도체 업황과 HBM(고대역폭메모리) 경쟁력**을 중심으로 ... 더 구체적으로 정리해드릴 수 있습니다.',
              additional_kwargs={'refusal': None},
              response_metadata={'token_usage': {'completion_tokens': 987, 'prompt_tokens': 17, 'total_tokens': 1004, ...}, 'model_name': 'gpt-5.6-terra', ..., 'finish_reason': 'stop', ...},
              id='lc_run--01a15317-...', tool_calls=[], invalid_tool_calls=[],
              usage_metadata={'input_tokens': 17, 'output_tokens': 987, 'total_tokens': 1004, ...}),
]}
'''
print(response["messages"][0])          # content='삼성전자의 주가를 예측해줘' additional_kwargs={} response_metadata={} id='...'
print(response["messages"][1])          # content='삼성전자 주가는 ...' additional_kwargs={'refusal': None} response_metadata={...} id='lc_run--...' ...
print(response["messages"][1].content)  # 답 문자열만 출력. 줄바꿈 문자가 실제 줄바꿈으로 나옴
'''
삼성전자 주가는 정확히 “예측”하기보다 **메모리 반도체 업황과 HBM(고대역폭메모리) 경쟁력**을 중심으로 시나리오별로 보는 것이 현실적입니다.
...
원하시면 현재 보유단가, 투자기간(단기·1년·3년 이상), 선호 위험도에 맞춰 **매수·보유·분할매도 전략 형태**로 더 구체적으로 정리해드릴 수 있습니다.
'''
print(response["messages"][0].type)     # human
print(response["messages"][1].type)     # ai
