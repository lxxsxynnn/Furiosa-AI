# 그래프 그림 저장 : lg05 ToolNode 그래프의 구조를 draw_mermaid_png로 PNG 파일로 저장
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
from langchain_tavily import TavilySearch

# 웹 검색 도구 생성
web_search = TavilySearch(
    max_results=3,
)

# result = web_search.invoke({
#     "query" : "삼성전자의 오늘 주가 알려줘"
# })

# print(result)

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

print("AI : ", response["messages"][-1].content)  # .content만 출력해서 메타데이터 없이 답 문자열만 나옴

############################################################
# 그래프 이미지 저장
############################################################
path = './_save/image/'    # 실행한 위치 기준 상대 경로. C:/study에서 실행해야 C:/study/_save/image/에 저장됨

png_data = graph.get_graph().draw_mermaid_png()   # 노드·엣지를 mermaid 그림으로 그린 PNG 바이트. 기본은 mermaid.ink 웹 API로 그려서 인터넷이 필요함
# print(graph.get_graph().draw_mermaid())          # 인터넷 없이 mermaid 문법 텍스트만 출력
'''
__start__ --> chatbot;
chatbot -.-> __end__;     # 점선 : 조건부 엣지 (tool_calls가 없을 때)
chatbot -.-> tools;       # 점선 : 조건부 엣지 (tool_calls가 있을 때)
tools --> chatbot;        # 실선 : add_edge
'''

with open(path + "lg07_ToolNode.png", "wb") as file:   # "wb" : 바이트(이미지)를 쓰기 모드로
    file.write(png_data)

# file = open(path + "lg07_ToolNode.png", "wb") > with open(path + "lg07_ToolNode.png", "wb")
# file.write(png_data)
# file.close() > with 블록이 끝나면 자동으로 close됨