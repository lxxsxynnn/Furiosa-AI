# ToolNode : 모델이 도구(웹 검색)를 쓰겠다고 하면 tools 노드에서 실행하고 결과를 들고 다시 chatbot으로
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
'''
{'query': '삼성전자의 오늘 주가 알려줘',
 'follow_up_questions': None,
 'answer': None,
 'images': [],
 'results': [
    {'url': 'https://kr.investing.com/equities/samsung-electronics-co-ltd',
     'title': 'Samsung Electronics Co Ltd 오늘의 주가 | 005930 실시간 티커 - Investing.com',
     'content': '### 주요 주주\n\n| 보유자 | 보유 비중 % | ... 오늘 삼성전자 주가는 218,000입니다. ... 적정가치',
     'score': 0.8646883, 'raw_content': None, 'id': 'c65c58-00'},
    {'url': 'https://ir.gsifn.io/sample/ir_search.html',
     'title': '주가검색 - 삼성전자',
     'content': '##### 삼성전자\n\n(2026/10/06)\n\n272,000 ... | 거래량 | 563,759 |',
     'score': 0.78886026, 'raw_content': None, 'id': '7850d4-01'},
    {'url': 'https://stock.mk.co.kr/price/home/KR7005930003',
     'title': '삼성전자(005930)- 매일경제 마켓',
     'content': '## 주가 정보\n\n 코스피\n\n  5,768.01 1.76% ... | 15:19:57 | 218,000 | 2,500 | 94 | 14,346,156 |',
     'score': 0.7520312, 'raw_content': None, 'id': '3b4cb3-02'},
 ],
 'response_time': 0.0,
 'request_id': 'e8dcdf2c-...'}
'''
# print(result['results'][-1]['content'])
'''
## 주가 정보

 코스피

  5,768.01 1.76%
...
## 삼성전자

(005930)    I     코스피  제조업 10.19 15:33 

| 218,000 | 전일 | 215,500 | 고가 | 219,500 | 상한가 | 280,000 | 거래량 (주) | 15,693,353 |
| 2,500 1.16% | 시가 | 207,000 | 저가 | 206,500 | 하한가 | 151,000 | 거래대금 (백만) | 3,374,908 |
...
| 15:19:55 | 218,000 | 2,500 | 23 | 14,345,648 |
'''

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
AI :  content='삼성전자(005930) 실시간 주가는 장중 계속 변동합니다.  \n정확한 오늘 시세는 [네이버페이 증권—삼성전자](https://finance.naver.com/item/main.naver?code=005930)에서 확인해 주세요.'
      additional_kwargs={'refusal': None}
      response_metadata={'token_usage': {'completion_tokens': 63, 'prompt_tokens': 2294, 'total_tokens': 2357, ...}, 'model_name': 'gpt-5.6-terra', ..., 'finish_reason': 'stop', ...}
      id='lc_run--01a1582b-...' tool_calls=[] invalid_tool_calls=[]
      usage_metadata={'input_tokens': 2294, 'output_tokens': 63, 'total_tokens': 2357, ...}
'''