from common import get_llm
import time

llm = get_llm(max_tokens=500)

prompt = "NPU의 특징을 말해줘"

times = []
for i in range(5):
    start = time.perf_counter()

    response = llm.invoke(prompt)

    end = time.perf_counter()

    elapsed= end - start

    times.append(elapsed)
    print(i + 1, "회: ", round(elapsed, 2), "초")

average = sum(times) / len(times)

print("평균: ", round(average, 2), "초")