# fan-out / fan-in : START에서 두 노드로 갈라져 같이 실행하고, 두 결과를 merge 노드에서 합치기
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. State 정의
class State(TypedDict):
    question : str
    search_result : str
    analysis_result : str
    final_result : str

# 2. 검색 노드
def search_agent(state: State):
    print("검색 에이전트 실행")
    return {"search_result" : "엔비디아는 AI GPU 시장의 주요기업이다."}

# 3. 분석 노드
def analysis_agent(state: State):
    print("분석 에이전트 실행")
    return {"analysis_result" : "AI 반도체 수요 증가가 성장 요인이다."}


# 4. 결과 통합 노드
def merge_agent(state: State):
    print("통합 노드 실행")
    result = (
        state["search_result"] + "\n" +
        state["analysis_result"]
    )
    return {"final_result" : result}

# 5. 그래프 구성
builder = StateGraph(State)
builder.add_node("search", search_agent)
builder.add_node("analysis", analysis_agent)
builder.add_node("merge", merge_agent)

################ 병렬 실행 ################
# fan-out : 한 노드에서 엣지 두 개가 나가면 두 노드가 같은 단계에서 같이 실행됨
builder.add_edge(START, "search")
builder.add_edge(START, "analysis")

################## 병합 ##################
# fan-in : 두 노드가 모두 끝난 다음 단계에서 merge가 한 번 실행됨
# search와 analysis가 서로 다른 키를 바꿔서 리듀서 없이도 충돌하지 않음. 같은 키를 동시에 바꾸면 InvalidUpdateError
builder.add_edge("search", "merge")
builder.add_edge("analysis", "merge")

builder.add_edge("merge", END)

# 6. 컴파일
graph = builder.compile()

# 실행
response = graph.invoke({
    "question" : "엔비디아의 미래 전망은?"
})
print(response["final_result"])
'''
검색 에이전트 실행
분석 에이전트 실행
통합 노드 실행
엔비디아는 AI GPU 시장의 주요기업이다.
AI 반도체 수요 증가가 성장 요인이다.
'''

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