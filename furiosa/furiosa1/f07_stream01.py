# f05 카피
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
)

model = client.models.list().data[0].id

# 추론
streaming = client.chat.completions.create(
    model = model,
    messages=[
        {
            "role" : "user",
            "content" : "Please tell me the difference between NPU and GPU in Korean.",
            # "content" : "NPU랑 GPU의 차이점을 설명해줘",
        }
    ],
    max_tokens=3000,
    stream=True,
)

for chunk in streaming:
    if chunk.choices:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)