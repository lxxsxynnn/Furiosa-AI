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
    "237241과 31321을 add로 더해줘"
)

# 5. 결과 확인
print("일반 응답: ", response.content)
print("====================================")
print("Tool Calls: ", response.tool_calls)  # Tool Calls:  [{'name': 'add', 'args': {'aaa': 237241, 'bbb': 31321}, 'id': 'call-6069f5ae816043dd82011a53506ce5ea', 'type': 'tool_call'}]
print("메타 데이터: ", response.response_metadata)
'''
메타 데이터:  {'token_usage': {'completion_tokens': 200, 'prompt_tokens': 176, 'total_tokens': 376, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cache_write_tokens': None, 'cached_tokens': 175}}, 'model_provider': 'openai', 'model_name': 'furiosa-ai/Qwen3-4B-FP8', 'system_fingerprint': None, 'id': 'chat-bf3c6438740548f99a3419c7b126791f', 'finish_reason': 'length', 'logprobs': None}
'''