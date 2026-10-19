# add_messages : 메시지 리스트를 합치는 리듀서 (id가 같으면 덮어씀)
from typing import Annotated, TypedDict
from langgraph.graph import add_messages

class MyData(TypedDict):    # 함수에서는 ()안에 인자가 파라미터지만 클래스에서는 상속을 의미
    messages : Annotated[list, add_messages]    # Reducer : (기존 값, 새 값)을 받아 합치는 함수. add_messages는 기존 리스트 뒤에 붙임

from langchain_core.messages import AIMessage, HumanMessage

msg1 = [HumanMessage(content='안녕', id='1')]
msg2 = [AIMessage(content='방가', id='2')]

results = add_messages(msg1, msg2)  # id가 다르면 뒤에 붙음
# print(results)  # [HumanMessage(content='안녕', additional_kwargs={}, response_metadata={}, id='1'), AIMessage(content='방가', additional_kwargs={}, response_metadata={}, id='2', tool_calls=[], invalid_tool_calls=[])]

msg3 = [HumanMessage(content='안녕', id='1')]
msg4 = [AIMessage(content='방가', id='1')]

results2 = add_messages(msg3, msg4)
print(results2)     # [AIMessage(content='방가', additional_kwargs={}, response_metadata={}, id='1', tool_calls=[], invalid_tool_calls=[])]
# id가 동일하면 뒤에 걸로 덮어써짐