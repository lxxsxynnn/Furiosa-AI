import os
from dotenv import load_dotenv

# pip install faiss-cpu
import faiss

# langchain에서 지원
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# FAISS에 저장한 데이터 불러와서 유사도 검색하기
load_dotenv()
api_key = os.environ['MONOROUTER_API_KEY'].strip()  # strip() : 앞뒤 공백·줄바꿈 제거
base_url = 'https://monogpt.kr/api/monorouter/v1'

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(      # 저장할 때와 같은 모델·dimensions여야 검색 가능 (Faiss17은 5차원)
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
    dimensions=5,           # text-embedding-3 계열에서만 사용 가능 / 계산은 줄지만 정보도 줄어듦
)

faiss_index = faiss.IndexFlatL2(len(embeddings.embed_query("Hello World!")))
faiss_index = faiss.IndexFlatL2(1536)
print("FAISS 인덱스 초기화 완료")

# 직접 만든 인덱스의 차원. load_local은 이걸 안 쓰고 저장된 인덱스(5차원)를 읽음
print(faiss_index.d)    # 1536

# faiss_db = FAISS(
#     embedding_function=embeddings,
#     index=faiss_index,              # 검색 대상
#     docstore=InMemoryDocstore(),    # 문서 원문·메타데이터를 id → Document dict로 보관
#     index_to_docstore_id={},        # 인덱스 행 번호 → docstore id
# )

# 저장된 문서의 갯수 확인
# print(faiss_db.index.ntotal)        # 0

################ 위 방법은 잘 사용하지 않는 방법 ################
# DB_PATH = 'C:/study/_db/Chroma11/'
# os.makedirs(DB_PATH, exist_ok=True)

DB_PATH = 'C:/study/_db/Faiss17/'
os.makedirs(DB_PATH, exist_ok=True)

# db = FAISS.from_documents(
#     documents=split_doc1 + split_doc2,
#     embedding=embeddings,
# )

# db.save_local(
#     folder_path=DB_PATH,
#     index_name='faiss_index17',
# )

db = FAISS.load_local(
    folder_path=DB_PATH,
    index_name='faiss_index17',
    embeddings=embeddings,
    allow_dangerous_deserialization=True,   # .pkl을 pickle로 읽어서 직접 허용해야 열림 (기본 False면 ValueError). 직접 저장한 파일만
)

# 불러온 DB의 벡터 차원 수와 저장된 벡터 개수
print(db.index.d)        # 5
print(db.index.ntotal)   # 18

print("===================================================")
# 문서 저장소 ID 확인 (인덱스 행 번호 0~17 → docstore id)
print(db.index_to_docstore_id)
'''
{
    0: 'a6d59038-2c6c-4b02-bc54-0b9571abf0a7',
    1: '45ac4497-afa0-410f-aa28-b19e72fdb8aa',
    ...
    17: 'd7e6ed8d-ce10-4e42-ac4e-1a415393df38',
}                       # 18개
'''

print("===================================================")
# 저장된 결과 확인
print(db.docstore._dict)
'''
{
    'a6d59038-2c6c-4b02-bc54-0b9571abf0a7': Document(
        id='a6d59038-2c6c-4b02-bc54-0b9571abf0a7',
        metadata={'source': 'C:/study/_data/rag_data/samsung_outlook.txt'},
        page_content='삼성전자 사업 전망\n\n삼성전자는 메모리 반도체, 파운드리 ... 사업별 흐름을 구분하는 편이 정확하다.',
    ),
    '45ac4497-afa0-410f-aa28-b19e72fdb8aa': Document(
        id='45ac4497-afa0-410f-aa28-b19e72fdb8aa',
        metadata={'source': 'C:/study/_data/rag_data/samsung_outlook.txt'},
        page_content='반도체 부문에서 주목할 변화는 인공지능 데이터센터의 확산이다. ... 매출 성장으로 연결할 가능성이 있다.',
    ),
    ...
    'd7e6ed8d-ce10-4e42-ac4e-1a415393df38': Document(
        id='d7e6ed8d-ce10-4e42-ac4e-1a415393df38',
        metadata={'source': 'C:/study/_data/rag_data/nvidia_outlook.txt'},
        page_content='게임용 GPU, 전문 시각화, 자동차 분야는 ... 교육용 자료이며 투자 권유가 아니다.',
    ),
}                       # 18개
'''

print("===================================================")
# 유사도 검색
aaa = db.similarity_search("삼성전자 창업주에 대해 알려줘", k=2)
print(aaa)
'''
[
    Document(
        id='fbe550ce-6896-456e-8175-aba5a91e7d4d',
        metadata={'source': 'C:/study/_data/rag_data/samsung_outlook.txt'},
        page_content='삼성전자의 중장기 전망은 인공지능 메모리의 경쟁력, 첨단 ... 위한 교육용 자료이며 투자 권유가 아니다.',
    ),
    Document(
        id='8b7e20b4-7c86-4d1d-80c9-99b98f63999b',
        metadata={'source': 'C:/study/_data/rag_data/samsung_outlook.txt'},
        page_content='파운드리는 고객사가 설계한 반도체를 위탁 생산하는 사업이다. ... 시점을 나누어 살펴볼 필요가 있다.',
    ),
]
'''