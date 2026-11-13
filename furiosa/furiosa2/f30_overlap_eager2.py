'''
furiosa-llm serve furiosa-ai/Qwen3-4B-FP8 \
    --devices npu:6 \
    --host 0.0.0.0 \
    --port 8000
'''
# --enable-overlap-scheduling: Furiosa-LLM 서빙 언진에서 CPU의 스케쥴링 작업과 NPU의 추론 연산을 겹쳐 실행하도록 하는 최적화 옵션
# 
# Continuous Batching: 여러 사용자의 요청을 동적으로 묶어 NPU에서 처리하는 기술
# Overlap Scheduling: NPU가 현재 배치를 처리하는 동안 CPU가 다음 배치를 준비하는 기술
# <-> Eager Scheduling: 순차처리에 유리(Default)
import time
import statistics
from common import get_llm
from concurrent.futures import ThreadPoolExecutor

llm = get_llm(max_tokens=100)

prompts = [
    f"NPU 추론 최적화 방법 {i}번을 설명해줘"
    for i in range(16)
]

def ask(prompt):
    start = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed = time.perf_counter() - start
    
    usage = response.usage_metadata or {}
    tokens = usage.get("output_tokens", 0)
    return elapsed, tokens

for workers in [1, 2, 4, 8]:

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executer:
        results = list(executer.map(ask, prompts))

    total_time = time.perf_counter() - start

    latencies = [r[0] for r in results]
    total_tokens = sum(r[1] for r in results)

    print(f"\n======== Workers: {workers}========")

    print("전체 출력 토큰: ", total_tokens)
    print("전체 소요 시간: ", round(total_time, 2), "초")
    print("요청 처리량: ", round(len(results) / total_time), "req/s")
    print("Output TPS: ", round(total_tokens / total_time, 2), "token/s")

'''
################ eager 모드 ################
======== Workers: 1========
전체 출력 토큰:  1600
전체 소요 시간:  19.27 초
요청 처리량:  1 req/s
Output TPS:  83.05 token/s

======== Workers: 2========
전체 출력 토큰:  1600
전체 소요 시간:  10.37 초
요청 처리량:  2 req/s
Output TPS:  154.27 token/s

======== Workers: 4========
전체 출력 토큰:  1600
전체 소요 시간:  5.68 초
요청 처리량:  3 req/s
Output TPS:  281.67 token/s

======== Workers: 8========
전체 출력 토큰:  1600
전체 소요 시간:  3.26 초
요청 처리량:  5 req/s
Output TPS:  490.42 token/s

################ overlap 모드 ################
전체 출력 토큰:  1600
전체 소요 시간:  18.38 초
요청 처리량:  1 req/s
Output TPS:  87.05 token/s

======== Workers: 2========
전체 출력 토큰:  1600
전체 소요 시간:  10.59 초
요청 처리량:  2 req/s
Output TPS:  151.12 token/s

======== Workers: 4========
전체 출력 토큰:  1600
전체 소요 시간:  5.54 초
요청 처리량:  3 req/s
Output TPS:  288.97 token/s

======== Workers: 8========
전체 출력 토큰:  1600
전체 소요 시간:  3.2 초
요청 처리량:  5 req/s
Output TPS:  500.17 token/s
'''