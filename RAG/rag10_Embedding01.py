# LCEL = LangChain Expression Language
# chain = prompt | model | output_parser

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

# 문장을 임베딩 벡터로 바꾸기
load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()  # strip() : 줄바꿈, 띄어쓰기 무시
base_url = 'https://monogpt.kr/api/monorouter/v1'

prompt = "삼성전자의 창업주는 누구인가요?"

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    # model="text-embedding-3-small",
    model="text-embedding-3-large",
    api_key=api_key,
    base_url=base_url,
)

# 벡터화
vector = embeddings.embed_query(prompt)
print(vector)
print('-------------------------------')
print("Dimension of Embedding Vector : ", len(vector))  # Dimension of Embedding Vector :  small - 1536 / large - 3072