import os
from dotenv import load_dotenv
from langchain_chroma import Chroma

# create_retrieval_chain으로 검색부터 답변까지 체인 하나로 묶기
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

########################## Retrievers ##########################
retriever = vector_store.as_retriever(search_kwargs={"k":2})
print(retriever)
# tags=['Chroma', 'OpenAIEmbeddings'] vectorstore=<langchain_chroma.vectorstores.Chroma object at 0x0000012988796C50> search_kwargs={'k': 2}

########################## Connect to Model ##########################
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model='gpt-5.6-luna',
    temperature=0,
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)

from langchain_core.prompts import ChatPromptTemplate
# langchain 1.x에는 chains가 없어서 예전 체인 함수는 langchain_classic에서 가져옴
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# 변수 이름은 {context}·{input}으로 고정 (context가 없으면 체인 만들 때 ValueError)
prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트를 바탕으로 질문에 답변해주세요. 컨텍스트 관련 정보가 없다면,
"주어진 정보로는 답변할 수 없습니다."라고 말씀해 주세요.

컨텍스트: {context}
질문: {input}
답변:
""")

# langchain 체인 생성
docu_chain = create_stuff_documents_chain(model, prompt)    # context의 문서들을 \n\n으로 이어 붙여 prompt에 넣고 prompt | model | StrOutputParser
rag_chain = create_retrieval_chain(retriever, docu_chain)   # input으로 찾은 문서를 context에, docu_chain 결과를 answer에 담은 dict를 돌려줌

# chain 실행
query = "삼성전자의 창업자는 누구인가요?"
response = rag_chain.invoke({"input" : query})

print(response)
'''
{
    'input': '삼성전자의 창업자는 누구인가요?',
    'context': [
        Document(
            id='4d391fcf-e448-4fdf-897d-1729e3234ae3',
            metadata={'source': 'C:/study/_data/rag_data\\samsung_outlook.txt'},
            page_content='삼성전자 사업 전망\n\n삼성전자는 메모리 반도체, 파운드리 ... 사업별 흐름을 구분하는 편이 정확하다.',
        ),
        Document(
            id='85966eba-f72d-4903-9d85-607e311cdd40',
            metadata={'source': 'C:/study/_data/rag_data\\samsung_outlook.txt'},
            page_content='삼성전자의 중장기 전망은 인공지능 메모리의 경쟁력, 첨단 ... 위한 교육용 자료이며 투자 권유가 아니다.',
        ),
    ],
    'answer': '주어진 정보로는 답변할 수 없습니다.',
}
'''
print(response.keys())      # dict_keys(['input', 'context', 'answer'])
print(response['answer'])   # 주어진 정보로는 답변할 수 없습니다.
