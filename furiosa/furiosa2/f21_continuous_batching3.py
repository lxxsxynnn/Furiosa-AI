import time
from concurrent.futures import ThreadPoolExecutor
from common import run_once

def ask(name, delay, max_tokens):
    time.sleep(delay)

    print(f"{name} 요청 시작", flush=True)

    start = time.perf_counter()
    result = run_once(
        f"NPU와 LLM 추론 최적화를 자세히 설명해줘. 요청 {name}",
        max_tokens,
    )
    elapsed = time.perf_counter() - start

    print(f"{name} 요청 완료: {elapsed:.2f}초", flush=True)

    return result

with ThreadPoolExecutor(max_workers=4) as pool: # with문 안에 넣으면 자동으로 닫힘
    futures = [
        pool.submit(ask, "A", 0,800),
        pool.submit(ask, "B", 1,200),
        pool.submit(ask, "C", 2,1000),
        pool.submit(ask, "D", 3,200),
        pool.submit(ask, "E", 4,300),
    ]
    results = [f.result() for f in futures]

# 위에랑 동일한 코드
# pool = ThreadPoolExecutor(max_workers=4)
# futures = [
#     pool.submit(ask, "A", 0,800),
#     pool.submit(ask, "B", 1,200),
#     pool.submit(ask, "C", 2,1000),
#     pool.submit(ask, "D", 3,200),
#     pool.submit(ask, "E", 4,300),
# ]
# results = [f.result() for f in futures]
# pool.shutdown()

print("============ END ============")

'''
A 요청 시작
B 요청 시작
C 요청 시작
D 요청 시작
B 요청 완료: 2.12초
D 요청 완료: 2.30초
E 요청 시작
A 요청 완료: 9.05초
E 요청 완료: 3.15초
C 요청 완료: 10.37초
============ END ============
'''