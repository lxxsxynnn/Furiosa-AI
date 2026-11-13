# time to first chunk
from common import get_llm
import time

llm = get_llm()

prompt = "NPU의 장점을 자세히 설명해줘"

start = time.perf_counter()

chunk = next(iter(llm.stream(prompt)))

first_time = time.perf_counter() - start

print("================================")
print("첫 컨텐츠 응답 시간: ", round(first_time, 2), "초")  # 첫 컨텐츠 응답 시간:  0.04 초