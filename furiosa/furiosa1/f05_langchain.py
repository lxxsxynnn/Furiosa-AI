# f01 카피
'''
furiosa-llm serve furiosa-ai/Qwern3-8B-FP8
furiosa-llm serve furiosa-ai/gpt-oss-20b
furiosa-llm serve furiosa-ai/Llama-3.1-8B-Instruct [X]
furiosa-llm serve furiosa-ai/Meta-Llama-3.1-8B-Instruct-FP8-dynamic
furiosa-llm serve furiosa-ai/Qwen3.5-0.8B [X]
furiosa-llm serve furiosa-ai/Qwen3-4B-FP8

furiosa-llm serve furiosa-ai/Qwen3-4B-FP8 \
    --devices npu:6 \
    --host 0.0.0.0 \
    --port 8000
'''

# pip install langchain-openai
# from openai import OpenAI
from langchain_openai import ChatOpenAI

# client = OpenAI(
#     base_url="http://127.0.0.1:8000/v1",
#     api_key="EMPTY",
# )

llm = ChatOpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
)

# 서버에서 로드된 모델 확인
# models = client.models.list()
# model_id = models.data[0].id

# print("Model: ", model_id)

# 추론
# response = client.chat.completions.create(
#     model = model_id,
#     messages=[
#         {
#             "role" : "user",
#             "content" : "Please tell me the difference between NPU and GPU in Korean.",
#             # "content" : "NPU랑 GPU의 차이점을 설명해줘",
#         }
#     ],
#     max_tokens=3000,
# )

response = llm.invoke("NPU와 GPU의 차이점을 설명해줘")

# print(response.choices[0].message.content)
print(response.content)