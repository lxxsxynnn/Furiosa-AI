import os
from dotenv import load_dotenv
from langchain_chroma import Chroma

# Chroma에 저장한 데이터 불러와서 retriever로 검색하기
load_dotenv()
api_key = os.environ['MONOROUTER_API_KEY'].strip()  # strip() : 앞뒤 공백·줄바꿈 제거
base_url = 'https://monogpt.kr/api/monorouter/v1'

# 임베딩
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
    # dimensions=5,
)

DB_PATH = 'C:/study/_db/Chroma12/'

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name='chroma12',
)
print(f"Number of Documents Saved in Vector DB: {vector_store._collection.count()}")    # Number of Documents Saved in Vector DB: 52

query = "삼성전자의 창업자는 누구인가요?"
result = vector_store.similarity_search(query)
print(f"Length of Results: {len(result)}")  # Length of Results: 4

########################## Retrievers ##########################
retriever = vector_store.as_retriever(search_kwargs={"k":2})
print(retriever)
# tags=['Chroma', 'OpenAIEmbeddings'] vectorstore=<langchain_chroma.vectorstores.Chroma object at 0x0000012988796C50> search_kwargs={'k': 2}
aaa = retriever.invoke(query)
print(f"Number of Relevant Documents Found: {len(aaa)}")
# Number of Relevant Documents Found: 2
print(f"Preview of the First Relevant Document: {aaa[0].page_content[:50]}")
'''
Preview of the First Relevant Document: 삼성전자 사업 전망

삼성전자는 메모리 반도체, 파운드리, 스마트폰, 디스플레이와 가전 사
'''