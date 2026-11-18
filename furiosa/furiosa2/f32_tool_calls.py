'''
furiosa-llm serve furiosa-ai/Qwen3-4B-FP8 \
    --devices npu:6 \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-overlap-scheduling \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
'''
from common import get_llm
from langchain_core.tools import tool

# 1. 도구 정의
@tool
def add(aaa: int, bbb: int) -> int:
    """ 두 정수를 더합니다. """
    return aaa + bbb

print(add.name)         # 함수명(add)
print(add.description)  # add 함수의 주석(두 정수를 더합니다.)
print(add.args)         # {'aaa': {'title': 'Aaa', 'type': 'integer'}, 'bbb': {'title': 'Bbb', 'type': 'integer'}}

# 2.llm 생성
llm = get_llm(max_tokens=500)

# 3. 도구 등록
llm_with_tools = llm.bind_tools([add])

# 4. 질문
response = llm_with_tools.invoke(
    "237241과 31321을 add로 더해줘"
)

# 5. 결과 확인
for call in response.tool_calls:
    print("tool 이름: ", call['name'])      # tool 이름:  add
    print("tool 매개변수: ", call['args'])  # tool 매개변수:  {'aaa': 237241, 'bbb': 31321}
    print("호출 Id: ", call['id'])          # 호출 Id:  call-8d94a17633e74430b70de6b60371cd61