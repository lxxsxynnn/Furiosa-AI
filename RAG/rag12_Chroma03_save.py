import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma

# 폴더의 txt 파일을 전부 불러와 Chroma에 저장하고 retriever로 검색하기
load_dotenv()
api_key = os.environ['MONOROUTER_API_KEY'].strip()  # strip() : 앞뒤 공백·줄바꿈 제거
base_url = 'https://monogpt.kr/api/monorouter/v1'

from glob import glob

path = 'C:/study/_data/rag_data/'

# 폴더에서 텍스트 파일 목록 가져오기
txt_files = glob(os.path.join(path, "*.txt"))
print(txt_files)
# ['C:/study/_data/rag_data\\2026_AI_for_All.txt', 'C:/study/_data/rag_data\\nvidia_outlook.txt', 'C:/study/_data/rag_data\\samsung_outlook.txt']

# 데이터 불러오기
data = []

for text_file in txt_files:
    loader = TextLoader(text_file, encoding="UTF-8",)
    data += loader.load()

# print(data)
print(len(data))            # 3
print(data[0].page_content)

char_count = [len(doc.page_content) for doc in data]
print(char_count)   # [8158, 2049, 1898]

# 분할 기준 만들기(Text Splitter)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=10,                      # 앞 조각 끝부분을 다음 조각 앞에 겹쳐 넣는 길이로, 경계에서 잘린 문장이 한 조각 안에 온전히 남게 해서 검색할 때 문맥을 놓치지 않게 함
    separators=["\n\n", "\n", " ", ""],    # 기본값과 같음 (생략 가능)
)

texts = text_splitter.split_documents(data)
print("생성된 텍스트 청크 수 : ", len(texts))   # 생성된 텍스트 청크 수 :  52
print("각 청크의 길이 : ", list(len(text.page_content) for text in texts))  # ( ... for ...)는 제너레이터라 그대로 print하면 <generator object>만 나옴. list()로 끝까지 꺼내야 값이 보임
'''
각 청크의 길이 :  [259, 282, 282, ..., 298, 282, 187, 249]
'''
# print("첫 번째 청크의 내용 : ", texts[0])
'''
page_content='2026년 한국의 AI for All 프로젝트와 생성형 AI 서비스 확산 ...'   # 청크 본문
metadata={'source': 'C:/study/_data/rag_data\\2026_AI_for_All.txt'}         # 원본 파일 경로
'''
# print("첫 번째 청크의 길이 : ", len(texts[0].page_content)) # 첫 번째 청크의 길이 :  259
# print("두 번째 청크의 내용 : ", texts[1])

# 임베딩
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
    # dimensions=5,
)

sample_text = "삼성전자의 창업자는 누구인가요?"
vector = embeddings.embed_query(sample_text)   # 문자열 하나는 embed_query → 벡터 하나 / 문자열 리스트는 embed_documents → 벡터 리스트
# print(vector)
print(len(vector))  # 1536

DB_PATH = 'C:/study/_db/Chroma12/'
os.makedirs(DB_PATH, exist_ok=True)

# 저장 - 실행할 때마다 같은 문서가 새 id로 추가됨 (두 번 실행하면 52개 > 104개)
vector_store = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
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