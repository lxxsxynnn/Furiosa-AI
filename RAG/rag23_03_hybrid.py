# 하이브리드 검색 : BM25와 벡터 검색 결과를 EnsembleRetriever로 합치기
# pip install rank_bm25
from langchain_core.documents import Document

texts = [
    Document(page_content="삼성전자는 반도체를 생산하는 기업이다."),
    Document(page_content="삼성전자는 갤럭시 스마트폰을 판매한다."),
    Document(page_content="SK하이닉스는 메모리 반도체를 생산한다."),
    Document(page_content="엔비디아는 인공지능 연산용 GPU를 설계한다."),
    Document(page_content="애플은 아이폰을 개발하는 기업이다."),
    Document(page_content="테슬라는 전기자동차를 제조한다."),
    Document(page_content="삼성전자는 AI 반도체 기술을 연구한다."),
    Document(page_content="반도체는 컴퓨터의 핵심 부품이다."),
]

query = "삼성전자의 반도체 사양"

######################## BM25 검색 ########################
# 키워드 기반 검색
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(texts)
bm25_retriever.k = 3

# bm25_docs = bm25_retriever.invoke(query)

# print("============ BM25 검색 결과 ============")
# for doc in bm25_docs:
#     print(doc.page_content)
'''
삼성전자는 AI 반도체 기술을 연구한다.
반도체는 컴퓨터의 핵심 부품이다.
테슬라는 전기자동차를 제조한다.
'''

import os
import gradio as gr
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

load_dotenv()
api_key = os.environ['MONOROUTER_API_KEY'].strip()
base_url = 'https://monogpt.kr/api/monorouter/v1'

# 임베딩
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={
        "device" : "cpu",
        # "local_files_only" : True,
    },
)

DB_PATH = 'C:/study/_db/Chroma12/'
import shutil
shutil.rmtree(DB_PATH, ignore_errors=True)   # 다시 실행하면 같은 문서가 중복으로 쌓여서 지우고 새로 만듦
os.makedirs(DB_PATH, exist_ok=True)

vector_store = Chroma.from_documents(
    embedding=embeddings,   # embeddings= 는 TypeError
    documents=texts,
    persist_directory=DB_PATH,
    collection_name='chroma12',
)

vector_retriever = vector_store.as_retriever(
    search_kwargs= {"k": 3}
)

# vector_docs = vector_retriever.invoke(query)

# print("============ 벡터 검색 결과 ============")
# for doc in vector_docs:
#     print(doc.page_content)
'''
============ 벡터 검색 결과 ============
삼성전자는 AI 반도체 기술을 연구한다.
삼성전자는 반도체를 생산하는 기업이다.
삼성전자는 갤럭시 스마트폰을 판매한다.
'''

######################## 하이브리드 검색 ########################
from langchain_classic.retrievers import EnsembleRetriever

hybrid_retriever = EnsembleRetriever(
    retrievers=[
        bm25_retriever, 
        vector_retriever,
    ],
    weights=[0.5, 0.5],     # 검색 중요도
)

hybrid_docs = hybrid_retriever.invoke(query)


print("============ 하이브리드 검색 결과 ============")
for doc in hybrid_docs:
    print(doc.page_content)

'''
============ 하이브리드 검색 결과 ============
삼성전자는 AI 반도체 기술을 연구한다.
반도체는 컴퓨터의 핵심 부품이다.
삼성전자는 반도체를 생산하는 기업이다.
테슬라는 전기자동차를 제조한다.
삼성전자는 갤럭시 스마트폰을 판매한다.
'''
# 4개 > 5개 : DB에 중복이 쌓였을 때는 벡터 결과가 AI 반도체 문서 2번 + 생산 문서라 4개만 나왔음
# BM25 3개 + 벡터 3개 중 겹치는 1개(AI 반도체)를 합쳐서 5개
# 점수 = weight / (순위 + 60)을 검색기마다 더함. 두 검색기 모두 1위인 문서가 맨 위