import os
from dotenv import load_dotenv
from langchain_chroma import Chroma

# retriever로 찾은 문서를 질문 앞에 붙여 모델에 넣기
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

# query = "삼성전자의 창업자는 누구인가요?"
query = "엔비디아의 주가가 급등한 계기는 몇 번 있었고 원인은 어떻게 되나요?"
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

########################## Connect to Model ##########################
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model='gpt-5-nano',
    temperature=0,
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)

# response = model.invoke("삼성전자의 창업자는 누구인가요?")
response = model.invoke("엔비디아의 주가가 급등한 계기는 몇 번 있었고 원인은 어떻게 되나요?")
print("model's response: ", response.content)
'''
삼성전자의 창업자는 이병철(또는 이건희의 할아버지로 창업자)이 설립한 삼성 그룹의 창업자 이병철입니다. 
삼성전자는 1969년에 설립된 삼성 그룹의 계열사로, 창업자 이병철이 1938년에 삼성상회를 설립하며 사업을 시작했고 이후 다각화되며 삼성전자는 1969년에 반도체·가전 부문으로 시작했습니다.
'''

query_with_context= f"""
    {aaa[0].page_content}\n\n
    위 내용에 근거해 다음 질문에 답변하세요.\n\n
    {query}
"""

response = model.invoke(query_with_context)
print("model's response of the query with context: ", response.content)
'''
삼성전자의 창업자는 이건희(Lee Kun-hee)의 선조인 이병철(이건희의 아버지)이 아닌가요? 
정확히 말하면 삼성그룹의 창립자는 이병철으로, 회사 자체를 설립한 인물입니다. 
삼성전자는 이병철의 삼성그룹 설립 이후 생겨난 계열사 중 하나로, 독립적으로 “창립자”를 가지는 회사로 보기는 어렵고 삼성그룹의 창립자인 이병철이 창립자로 간주됩니다. 
이후 삼성전자는 이건희 시대를 거치며 현재의 글로벌 대기업으로 성장했습니다.
'''