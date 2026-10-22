# fan-out / fan-in + LLM 통합 : 웹 검색·LLM 분석 결과를 merge 노드에서 LLM에 넘겨 하나의 답으로 정리
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
    max_tokens=4000,    # 1000 > 4000 : 1000일 때는 merge의 긴 프롬프트에서 출력 토큰이 모자라 답이 빈 문자열이었음
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

def merge_agent(state: State):      # 문자열 이어 붙이기 > LLM에 두 결과를 넘겨 겹치는 내용을 정리하고 하나의 답으로
    # print("통합 노드 실행")
    # result = (
    #     state["search_result"] + "\n\n" +
    #     "====================================" + "\n" +
    #     state["analysis_result"]
    # )
    # return {"messages" : [("assistant", result)]}  # message > messages : State에 없는 키라 답이 messages에 안 붙었음 / ("assistant", 문자열)은 add_messages가 AIMessage로 바꿈
    
    print("결과 통합 에이전트 실행")

    prompt = f"""
다음 두 결과를 종합하여 최종 답변을 작성해줘

[웹 검색 결과]
{state["search_result"]}

[AI 분석 결과]
{state["analysis_result"]}

검색 결과를 우선적으로 참고하고,
최신 정보와 기존 분석을 구분해서 설명해줘
"""

    response = llm.invoke(prompt)   # 문자열을 넣으면 HumanMessage 하나로 보냄

    return {"messages" : [response]}    # response가 이미 AIMessage라 그대로 messages에 붙임

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
# search와 analysis가 서로 다른 키를 바꿔서 리듀서 없이도 충돌하지 않음. 같은 키를 동시에 바꾸면 InvalidUpdateError
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
'''
Assistant :  ## 엔비디아(NVIDIA) 미래 전망 종합
엔비디아의 미래는 전반적으로 **AI 인프라 시장 확대의 핵심 수혜자**라는 점에서 긍정적으로 평가할 수 있습니다. ...
아래에서는 제공된 **웹 검색 결과에 담긴 최신 시점 정보**와, 이를 해석한 **기존 AI 분석**을 구분해 정리합니다.

## 1. 웹 검색 결과 기준: 최근 시장이 보는 엔비디아     <- search_result
라이트파이낸스 자료는 2026년 10월 17일 기준 엔비디아 주가를 214.11달러 ... 2026년 말 애널리스트 전망 범위는 대략 164.75~284달러
- 낙관론 / 신중론
- 핵심 성장 분야 : 데이터센터와 클라우드 AI, 자율주행 플랫폼, 의료 AI ... 로보틱스 및 엣지 AI

## 2. 기존 분석: 엔비디아가 강한 이유                  <- analysis_result
### ① AI 가속기 시장의 선도 지위 (CUDA 생태계 ...)
### ② 빅테크의 AI 설비투자
### ③ 신제품과 신규 사업 확장

## 3. 주의해야 할 위험 요인
### ① 높은 기대치와 밸류에이션 부담
### ② 경쟁사와 자체 AI 칩       <- "검색 결과는 AMD·인텔·퀄컴 등을 주요 경쟁사로 언급합니다" + 기존 지식(구글 TPU 등)
### ③ 미·중 규제와 중국 시장
### ④ 공급망 의존과 마진 정상화

## 4. 향후 시나리오
| 시나리오 | 전개 가능성 | 주요 조건 |   (낙관적 / 기준 / 비관적)

## 5. 앞으로 확인할 핵심 지표
1. 데이터센터 매출 성장률
...
7. AI 서비스의 수익화 진전

## 결론
... 엔비디아는 **중장기 성장 가능성은 높지만, 단기 변동성도 큰 종목**으로 보는 것이 합리적입니다. ...
'''
# 프롬프트대로 1장은 검색 결과(최신 수치), 2장은 기존 분석으로 나눠서 씀. 3장부터는 둘을 섞어서 씀

# 그래프 이미지 그리기
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

png_data = graph.get_graph().draw_mermaid_png()   # 노드·엣지를 mermaid 그림으로 그린 PNG 바이트. 기본은 mermaid.ink 웹 API로 그려서 인터넷이 필요함

img = Image.open(BytesIO(png_data))     # BytesIO : 바이트를 파일처럼 읽게 감싸줌. 디스크에 저장하지 않고 PIL 이미지로 열기

# 그림 출력
plt.figure(figsize=(12, 8))
plt.imshow(img)     # Jupyter 환경에서는 여기까지만 해도 나옴
plt.axis('off')
plt.show()