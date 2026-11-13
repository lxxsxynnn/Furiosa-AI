'''
furiosa-llm serve furiosa-ai/Qwen3-4B-FP8 \
    --devices npu:6 \
    --host 0.0.0.0 \
    --port 8000
'''

import time
from common import get_llm

llm = get_llm(max_tokens=100)

prompts = [
    f"NPU 추론 최적화 방법 {i}번을 설명해줘"
    for i in range(8)
]

start = time.perf_counter()
total_tokens = 0

for prompt in prompts:
    response = llm.invoke(prompt)

    usage = response.usage_metadata or {}
    tokens = usage.get("output_tokens", 0)

    total_tokens += tokens

elapsed = time.perf_counter() - start

print("전체 요청; ", len(prompts))
print("전체 출력 토큰: ", total_tokens)
print("전체 소요시간: ", round(elapsed, 3), "초")
print("RPS: ", round(len(prompts) / elapsed, 3))
print("TPS: ", round(total_tokens / elapsed, 3))

'''
################ eager 모드 ################
전체 요청;  8
전체 출력 토큰:  800
전체 소요시간:  9.84 초
RPS:  0.813
TPS:  81.297

터미널 들어가보면
2026-11-13T04:16:21.13306418Z  INFO furiosa_generator::next_gen::generator: Eager scheduler has started with:
나와있음
>> 여태 Eager 모드로 사용 중이었다는 뜻

################ overlab 모드 ################
서버에서 아래 명령어 실행
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 --devices npu:6 --host 0.0.0.0 --port 8000 --enable-overlap-scheduling

전체 요청;  8
전체 출력 토큰:  800
전체 소요시간:  9.974 초
RPS:  0.802
TPS:  80.21
'''