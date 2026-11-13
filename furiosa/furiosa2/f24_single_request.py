import time
from common import get_llm

llm = get_llm(max_tokens=100)

start = time.perf_counter()
response = llm.invoke("NPU의 장점을 세 가지 설명해줘.")
elapsed = time.perf_counter() - start

usage = response.usage_metadata or {}

print("응답: ", response.content)
print("지연시간: ", round(elapsed, 3), "초")
print("입력 토큰: ", usage.get("input_tokens", 0))
print("출력 토큰: ", usage.get("output_tokens", 0))

'''
응답:  <think>
Okay, the user is asking for three advantages of NPU (Neural Processing Unit). Let me start by recalling what an NPU is. It's a specialized chip designed for machine learning tasks, right? So, the main points would be about performance, efficiency, and maybe integration with other systems.

First, I need to think about the primary benefits. High performance in AI tasks is a big one. NPUs are optimized for parallel processing, which makes them faster than general-purpose CPUs
지연시간:  0.904 초
입력 토큰:  19
출력 토큰:  100
'''