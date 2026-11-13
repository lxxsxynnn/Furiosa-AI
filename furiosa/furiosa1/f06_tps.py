# f05 카피
from langchain_openai import ChatOpenAI
import time

max_tokens = 1000

llm = ChatOpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
    max_tokens=max_tokens,
    temperature=2,
    top_p=1,
)

start = time.perf_counter()
response = llm.invoke("NPU와 GPU의 차이점을 설명해줘")
end = time.perf_counter()

elapsed = round(end - start, 2)

# print(response.content)
print("추론 시간: ", elapsed, "초")
print("출력 토큰: ", max_tokens)
print("평균 처리량: ", round(max_tokens/elapsed, 2), " tokens/s")

'''
추론 시간:  7.17 초
출력 토큰:  1000
평균 처리량:  139.47  tokens/s
'''