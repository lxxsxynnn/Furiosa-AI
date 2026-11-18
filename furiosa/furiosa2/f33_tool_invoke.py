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
llm = get_llm(max_tokens=200)

# 3. 도구 등록
llm_with_tools = llm.bind_tools([add])

# 4. 질문
response = llm_with_tools.invoke(
    "21과 32를 add로 더해줘"
)

# 5. 결과 확인
for call in response.tool_calls:
    result = add.invoke(call['args'])
    print("계산 결과: ", result)

# 계산 결과:  53