# 11-1 카피
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# pip install faiss-cpu
import faiss

# langchain에서 지원
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# 데이터 벡터화해서 벡터 DB(FAISS)에 넣기
load_dotenv()
api_key = os.environ['MONOROUTER_API_KEY'].strip()  # strip() : 앞뒤 공백·줄바꿈 제거
base_url = 'https://monogpt.kr/api/monorouter/v1'

# 데이터 불러오기
path = 'C:/study/_data/rag_data/'
loader1 = TextLoader(path + 'samsung_outlook.txt', encoding='UTF-8')
loader2 = TextLoader(path + 'nvidia_outlook.txt', encoding='UTF-8')

# 분할 기준 만들기(Text Splitter)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,                      # 앞 조각 끝부분을 다음 조각 앞에 겹쳐 넣는 길이로, 경계에서 잘린 문장이 한 조각 안에 온전히 남게 해서 검색할 때 문맥을 놓치지 않게 함
    separators=["\n\n", "\n", " ", ""],     # 기본값과 같음 (생략 가능)
)

# 문서 분할(chunking)
split_doc1 = loader1.load_and_split(text_splitter) # chunk 300, overlap 100
split_doc2 = loader2.load_and_split(text_splitter)

print(len(split_doc1))  # 9
print(len(split_doc2))  # 9
print(split_doc1)

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
    dimensions=5,           # text-embedding-3 계열에서만 사용 가능 / 계산은 줄지만 정보도 줄어듦
)

faiss_index = faiss.IndexFlatL2(len(embeddings.embed_query("Hello World!")))
faiss_index = faiss.IndexFlatL2(1536)
print("FAISS 인덱스 초기화 완료")

# 직접 만든 인덱스의 차원. from_documents는 이걸 안 쓰고 임베딩 길이(5)로 새로 만듦
print(faiss_index.d)    # 1536

# faiss_db = FAISS(
#     embedding_function=embeddings,
#     index=faiss_index,              # 검색 대상
#     docstore=InMemoryDocstore(),    # 문서 원문·메타데이터를 id → Document dict로 보관
#     index_to_docstore_id={},        # 인덱스 행 번호 → docstore id
# )

# 저장된 문서의 갯수 확인
# print(faiss_db.index.ntotal)        # 0

##########################################################
# DB_PATH = 'C:/study/_db/Chroma11/'
# os.makedirs(DB_PATH, exist_ok=True)

# db = Chroma.from_documents(
#     documents=split_doc1 + split_doc2,
#     embedding=embeddings,
#     persist_directory=DB_PATH,
#     collection_name='chroma11',
# )
# print('Successfully Saved Chroma Docs')

DB_PATH = 'C:/study/_db/Faiss17/'
os.makedirs(DB_PATH, exist_ok=True)

# 문서를 임베딩해서 인덱스를 새로 만들고 넣음 (위 faiss_index는 안 씀)
db = FAISS.from_documents(
    documents=split_doc1 + split_doc2,
    embedding=embeddings,
)

# faiss_index17.faiss(벡터)와 .pkl(문서·id 매핑) 두 파일로 저장. 다시 실행하면 덮어써서 18개 그대로
db.save_local(
    folder_path=DB_PATH,
    index_name='faiss_index17',
)