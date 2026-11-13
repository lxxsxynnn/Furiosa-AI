# 동시 요청 수별 비교
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
======== Workers: 1========
전체 출력 토큰:  1600
전체 소요 시간:  14.29 초
요청 처리량:  1 req/s
Output TPS:  111.99 token/s

======== Workers: 2========
전체 출력 토큰:  1600
전체 소요 시간:  8.65 초
요청 처리량:  2 req/s
Output TPS:  184.92 token/s

======== Workers: 4========
전체 출력 토큰:  1600
전체 소요 시간:  4.53 초
요청 처리량:  4 req/s
Output TPS:  353.51 token/s

======== Workers: 8========
전체 출력 토큰:  1600
전체 소요 시간:  2.61 초
요청 처리량:  6 req/s
Output TPS:  612.8 token/s
'''