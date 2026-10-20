# DuckDuckGo 검색 도구 : API 키 없이 쓰는 웹 검색을 ToolNode에 연결
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
# pip install langchain-tavily
# .env에 TAVILY_API_KEY가 있어야 함
# from langchain_tavily import TavilySearch

# # 웹 검색 도구 생성
# web_search = TavilySearch(
#     max_results=3,
# )
# pip install -U ddgs   # API 키가 필요 없음
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
web_search = DuckDuckGoSearchRun(
    api_wrapper=DuckDuckGoSearchAPIWrapper(max_results=5),  # DuckDuckGoSearchRun(max_results=5)는 에러 없이 무시되고 5개 고정. 개수는 api_wrapper에
)

# result = web_search.invoke({
#     "query" : "삼성전자의 오늘 주가 알려줘"
# })

# print(result)
# exit()

tools = [web_search]

llm_with_tools = llm.bind_tools(tools)  # 모델에 도구 이름·설명·인자를 알려줌. 모델은 도구를 직접 실행하지 않고 tool_calls만 만듦

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

###################### 1. State 정의 ######################
class State(TypedDict):
    messages : Annotated[list, add_messages]

###################### 2. Node 정의 ######################
def chatbot(state: State):     # 답 대신 tool_calls가 든 AIMessage가 나올 수 있음
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

################ 3. Graph 구성, Node 추가 ################
# 그래프 생성
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)

# tool_node = ToolNode(tools)
# graph_builder.add_node("tools", tool_node)
graph_builder.add_node("tools", ToolNode(tools))    # tool_calls대로 도구를 실행해 결과를 ToolMessage로 붙임. tools_condition이 "tools"를 돌려줘서 이름을 tools로

###################### 4. Edge 추가 ######################
graph_builder.add_edge(START, "chatbot")

# graph_builder.add_edge("chatbot", "tools")
graph_builder.add_conditional_edges(    # 조건부 엣지 : 함수가 돌려준 노드 이름으로 이동
    "chatbot",
    tools_condition     # 마지막 메시지의 tool_calls 존재 여부 확인
    # tool_calls가 있으면 'tools' 반환
    # tool_calls가 없으면 END('__end__') 반환 > 따로 add_edge("chatbot", END)를 안 해도 됨
)

graph_builder.add_edge("tools", "chatbot")  # 검색 결과를 들고 다시 모델로. tool_calls가 없을 때까지 chatbot ↔ tools 반복

######################## 5. 컴파일 ########################
graph = graph_builder.compile()

###################### 6. 그래프 실행 ######################
query = "삼성전자의 오늘 주가를 알려줘"

response = graph.invoke({
    "messages" : [("user", query)]
})

print("AI : ", response["messages"][-1])    # 마지막 메시지(최종 답)만 출력
'''
AI :  content='삼성전자 보통주(005930)의 현재 확인 가격은 **주당 약 218,000원**입니다.  \n장중 가격은 실시간으로 변동하므로, 정확한 현재가는 한국거래소·네이버페이 증권·증권사 HTS/MTS에서 확인해 주세요.'
      additional_kwargs={'refusal': None}
      response_metadata={'token_usage': {'completion_tokens': 71, 'prompt_tokens': 801, 'total_tokens': 872, ...}, 'model_name': 'gpt-5.6-terra', ..., 'finish_reason': 'stop', ...}
      id='lc_run--01a15859-...' tool_calls=[] invalid_tool_calls=[]
      usage_metadata={'input_tokens': 801, 'output_tokens': 71, 'total_tokens': 872, ...}
'''