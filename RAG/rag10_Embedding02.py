# LCEL = LangChain Expression Language
# chain = prompt | model | output_parser

import os
from dotenv import load_dotenv

# 임베딩 벡터의 dimension 조절하기
load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()  # strip() : 줄바꿈, 띄어쓰기 무시
base_url = 'https://monogpt.kr/api/monorouter/v1'

prompt = "삼성전자의 창업주는 누구인가요?"

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
    dimensions=5,           # text-embedding-3 계열에서만 사용 가능 / 계산은 줄지만 정보도 줄어듦
)

# 벡터화
vector = embeddings.embed_query(prompt)
print(vector)
print('-------------------------------')
print("Dimension of Embedding Vector : ", len(vector))  # Dimension of Embedding Vector :  5