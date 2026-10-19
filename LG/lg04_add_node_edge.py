# 노드 여러 개 잇기 : prepare → chatbot → finish를 add_edge로 차례로 연결한 그래프
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
from langchain_core.messages import AIMessage

################ 1. State 정의 ################
class State(TypedDict):
    messages : Annotated[list, add_messages]    # 노드가 돌려준 메시지를 기존 리스트 뒤에 붙임
    aaa : str   # 리듀서가 없는 키는 노드가 돌려준 값으로 덮어씀
    bbb : str
    ccc : str

################ 2. Node 정의 ################
def prepare(state: State):     # 모델 없이 정해 둔 메시지만 붙임
    return {
        "messages": [AIMessage(content="질문을 접수했습니다.")]
    }

def chatbot(state: State):     # 앞 노드가 붙인 '질문을 접수했습니다.'까지 대화 기록으로 모델에 넘어감
    return {"messages" : [llm.invoke(state["messages"])]}

def finish(state: State):      # 모델 답 뒤에 완료 메시지를 붙임
    return {
            "messages": [AIMessage(content="답변 처리가 완료되었습니다.")]
    }

################ 3. Graph 구성 ################
graph_01 = StateGraph(State)

graph_01.add_node("prepare", prepare)
graph_01.add_node("chatbot", chatbot)
graph_01.add_node("finish", finish)

################ 4. Edge 추가 ################
graph_01.add_edge(START, "prepare")     # START → prepare → chatbot → finish → END
graph_01.add_edge("prepare", "chatbot")
graph_01.add_edge("chatbot", "finish")
graph_01.add_edge("finish", END)

################ 5. Graph 컴파일 ################
graph = graph_01.compile()

################ 6. Graph 실행 ################
query = "삼성전자의 주가를 예측해줘"

response = graph.invoke({
        "messages" : [("user", query)]  # ("user", 문자열) 튜플은 add_messages가 HumanMessage로 바꿈
})

################ 7. 결과 출력 ################
for i, msg in enumerate(response['messages']):
    print(f"[{i}] {msg.type} : {msg.content}")     # 노드가 실행된 순서대로 메시지가 쌓임
'''
[0] human : 삼성전자의 주가를 예측해줘
[1] ai : 질문을 접수했습니다.
[2] ai : 삼성전자 주가를 정확히 예측하는 것은 불가능합니다. ... 투자 판단은 분산투자와 손절 기준을 함께 세우는 것이 좋습니다.
[3] ai : 답변 처리가 완료되었습니다.
'''