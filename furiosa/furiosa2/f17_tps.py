from common import get_llm
import time

llm = get_llm(max_tokens=1000)

prompt = "NPU의 특징을 자세히 설명해줘"

start = time.perf_counter()

response = llm.invoke(prompt)

elapsed = time.perf_counter() - start

tokens = response.usage_metadata["output_tokens"]

tps = tokens / elapsed

print("출력 토큰: ", tokens)    # 출력 토큰:  1000
print("전체 시간: ", round(elapsed, 2), "초")   # 전체 시간:  8.95 초
print("TPS: ", round(tps, 2))   # TPS:  111.7