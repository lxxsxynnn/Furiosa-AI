import os
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# FAISS에 저장한 pdf 청크 불러와서 챗봇 구현
load_dotenv()
api_key = os.environ['MONOROUTER_API_KEY'].strip()
base_url = 'https://monogpt.kr/api/monorouter/v1'

# 임베딩
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
)

# 저장된 데이터 불러오기
DB_PATH = 'C:/study/_db/Faiss19/'
os.makedirs(DB_PATH, exist_ok=True)

vector_store = FAISS.load_local(
    folder_path=DB_PATH,
    index_name='faiss_index19',
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)

# 검색기 생성
retriever = vector_store.as_retriever(search_kwargs={"k":2})

# 모델 생성
model = ChatOpenAI(
    model='gpt-5.6-luna',
    temperature=0,
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)

# 프롬프트 생성
prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트를 바탕으로 질문에 답변해주세요. 컨텍스트 관련 정보가 없다면,
"주어진 정보로는 답변할 수 없습니다."라고 말씀해 주세요.

컨텍스트: {context}
질문: {input}
답변:
""")

# 체인 생성
docu_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, docu_chain)

# 답변 생성 함수 정의
# ChatInterface가 (입력한 메시지, 이전 대화 리스트)로 부름. history는 체인에 안 넘겨서 질문마다 따로 검색함
def answer_invoke(message, history):
    response = rag_chain.invoke({"input" : message})
    return response['answer']

# 챗봇 인터페이스 정의
demo = gr.ChatInterface(fn=answer_invoke, title="YOLO")

# 챗봇 실행
demo.launch()