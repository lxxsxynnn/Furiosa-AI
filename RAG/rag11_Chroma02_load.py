import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma

# 벡터 DB(Chroma)에 저장된 데이터 불러와서 검색하기
load_dotenv()
api_key = os.environ['MONOROUTER_API_KEY'].strip()  # strip() : 앞뒤 공백·줄바꿈 제거
base_url = 'https://monogpt.kr/api/monorouter/v1'

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(      # 저장할 때와 같은 모델·dimensions여야 검색 가능
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
    dimensions=5,           # text-embedding-3 계열에서만 사용 가능 / 계산은 줄지만 정보도 줄어듦
    # text-embedding-3-small의 기본 dimension은 1536 (large는 3072)
)

DB_PATH = 'C:/study/_db/Chroma11/'
os.makedirs(DB_PATH, exist_ok=True)

# 불러오기
db = Chroma(
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name='chroma11',
)

# 저장된 데이터 확인
print("=============================================")
print(db.get())
'''
{
    'ids': [
        'fc0c6da5-7d4d-4393-9f40-831954a7270e',
        'df81d2b3-8574-4883-9ca3-42b6ae349cef',
        ...
        'b08458e8-a0eb-43e8-9819-b81d592c2c1c',
    ],                  # 36개
    'embeddings': None,                 # get()은 기본으로 metadatas·documents만 돌려줌. 벡터는 db.get(include=['embeddings'])
    'documents': [
        '삼성전자 사업 전망\n\n삼성전자는 메모리 반도체, 파운드리 ... 사업별 흐름을 구분하는 편이 정확하다.',
        '반도체 부문에서 주목할 변화는 인공지능 데이터센터의 확산이다. ... 매출 성장으로 연결할 가능성이 있다.',
        ...
        '게임용 GPU, 전문 시각화, 자동차 분야는 ... 교육용 자료이며 투자 권유가 아니다.',
    ],                  # 36개
    'uris': None,
    'included': ['metadatas', 'documents'],
    'data': None,
    'metadatas': [
        {'source': 'C:/study/_data/rag_data/samsung_outlook.txt'},
        ...
        {'source': 'C:/study/_data/rag_data/nvidia_outlook.txt'},
    ],                  # 36개
}
'''

# 질문 문장을 저장할 때와 같은 임베딩 모델로 벡터로 바꾼 뒤, DB에 저장된 조각 벡터 중 가장 가까운 k개(기본값 4)를 Document 리스트로 돌려줌
print("=============================================")
aaa = db.similarity_search("삼성전자 사업전망에 대해 알려줘", k=4)  # default: 4
print(aaa)
'''
[
    Document(
        id='030fe343-5c8a-497d-8878-c045623b07ac',
        metadata={'source': 'C:/study/_data/rag_data/samsung_outlook.txt'},
        page_content='기능이 개선되더라도 소비자가 새 기기로 바꿀 이유가 부족하면 ... 신제품 출시도 수익성에 영향을 준다.',
    ),
    Document(
        id='cdbd2ec7-b5fb-4b5b-8e61-2ceaf4589167',
        metadata={'source': 'C:/study/_data/rag_data/samsung_outlook.txt'},
        page_content='기능이 개선되더라도 소비자가 새 기기로 바꿀 이유가 부족하면 ... 신제품 출시도 수익성에 영향을 준다.',
    ),
    Document(
        id='322acaac-dc2a-4320-8da4-7164492ea7c9',
        metadata={'source': 'C:/study/_data/rag_data/samsung_outlook.txt'},
        page_content='삼성전자의 중장기 전망은 인공지능 메모리의 경쟁력, 첨단 ... 위한 교육용 자료이며 투자 권유가 아니다.',
    ),
    Document(
        id='079c3a78-f20b-41e1-880b-fb11797d41cc',
        metadata={'source': 'C:/study/_data/rag_data/samsung_outlook.txt'},
        page_content='삼성전자의 중장기 전망은 인공지능 메모리의 경쟁력, 첨단 ... 위한 교육용 자료이며 투자 권유가 아니다.',
    ),
]
'''