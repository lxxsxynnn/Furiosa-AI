# fan-out / fan-in 실전 : 같은 질문을 웹 검색과 LLM 분석에 동시에 보내고 두 결과를 합쳐 답으로
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

######################## 0. Tool 정의 ########################
from langchain_tavily import TavilySearch

# 웹 검색 도구 생성
web_search = TavilySearch(
    max_results=3,
)

# result = web_search.invoke({
#     "query" : "삼성전자의 오늘 주가 알려줘"
# })

# print(result)

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

###################### 1. State 정의 ######################
class State(TypedDict):
    messages : Annotated[list, add_messages]    # 질문과 최종 답
    search_result : str     # search 노드만 바꾸는 키
    analysis_result : str   # analysis 노드만 바꾸는 키

###################### 2. Node 정의 ######################
# 검색 에이전트
def search_agent(state: State):
    print("검색 에이전트 실행")
    question = state["messages"][-1].content    # 마지막 메시지 = 사용자 질문

    result = web_search.invoke({    # 모델이 tool_calls를 만드는 과정 없이 노드에서 도구를 바로 실행
        "query": question
    })

    return {"search_result" : str(result)}  # Tavily 결과는 dict라서 str 키에 맞게 문자열로

def analysis_agent(state: State):
    print("분석 에이전트 실행")
    question = state["messages"][-1].content

    response = llm.invoke(  # 도구 없는 llm이라 검색 없이 학습한 지식으로만 답함
        f"다음 질문을 기존 지식을 바탕으로 분석해줘\n{question}"
    )
    return {"analysis_result" : response.content}

def merge_agent(state: State):      # 단순 concat > prompt화하면 좀 더 성능이 좋은 에이전트가 나올 수 있음
    print("통합 노드 실행")
    result = (
        state["search_result"] + "\n\n" +
        "====================================" + "\n" +
        state["analysis_result"]
    )
    return {"messages" : [("assistant", result)]}  # message > messages : State에 없는 키라 답이 messages에 안 붙었음("assistant", 문자열)은 add_messages가 AIMessage로 바꿈

################ 3. Graph 구성, Node 추가 ################
# 그래프 생성
builder = StateGraph(State)
builder.add_node("search", search_agent)
builder.add_node("analysis", analysis_agent)
builder.add_node("merge", merge_agent)

################ 병렬 실행 ################
# fan-out : 한 노드에서 엣지 두 개가 나가면 두 노드가 같은 단계에서 같이 실행됨
builder.add_edge(START, "search")
builder.add_edge(START, "analysis")

################## 4. 병합 ##################
# fan-in : 두 노드가 모두 끝난 다음 단계에서 merge가 한 번 실행됨
# search와 analysis가 서로 다른 키를 바꿔서 리듀서 없이도 충돌하지 않음(같은 키를 동시에 바꾸면 InvalidUpdateError)
builder.add_edge("search", "merge")
builder.add_edge("analysis", "merge")

builder.add_edge("merge", END)

################ 5. 컴파일 ################
graph = builder.compile()

################ 6. 그래프 실행 ################
query = "엔비디아의 미래 전망에 대해 알려줘 "

response = graph.invoke(
    {"messages": [("user", query)]}
)

print("Assistant : ", response["messages"][-1].content)

# 그래프 이미지 그리기
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

png_data = graph.get_graph().draw_mermaid_png()   # 노드·엣지를 mermaid 그림으로 그린 PNG 바이트
# 기본은 mermaid.ink 웹 API로 그려서 인터넷이 필요함

img = Image.open(BytesIO(png_data))     # BytesIO : 바이트를 파일처럼 읽게 감싸줌, 디스크에 저장하지 않고 PIL 이미지로 열기

# 그림 출력
plt.figure(figsize=(12, 8))
plt.imshow(img)     # Jupyter 환경에서는 여기까지만 해도 나옴
plt.axis('off')
plt.show()