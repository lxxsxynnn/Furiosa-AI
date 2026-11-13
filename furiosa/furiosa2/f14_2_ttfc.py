# time to first chunk
from common import get_llm
import time

llm = get_llm()

prompt = "NPU의 장점을 자세히 설명해줘"

start = time.perf_counter()

first_time = None

for chunk in llm.stream(prompt):

    if chunk.content:
        if first_time is None:
            first_time = time.perf_counter() - start
            break

        print(chunk.content, end="", flush=True)    # 첫 번째 청크 출력

print("================================")
print("첫 컨텐츠 응답 시간: ", round(first_time, 2), "초")  # 첫 컨텐츠 응답 시간:  0.05 초