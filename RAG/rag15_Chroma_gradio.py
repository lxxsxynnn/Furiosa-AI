import os
from dotenv import load_dotenv
from langchain_chroma import Chroma

# Gradio 챗봇 실습(Chroma)
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


########################## Gradio 챗봇 ##########################
import gradio as gr

# ChatInterface가 (입력한 메시지, 이전 대화 리스트)로 부름. history는 체인에 안 넘겨서 질문마다 따로 검색함
def answer_invoke(message, history):
    response = rag_chain.invoke({"input" : message})
    return response['answer']

# Gradio 인터페이스 만들기
demo = gr.ChatInterface(fn=answer_invoke, title="Chatbot")

# Gradio 실행
demo.launch()