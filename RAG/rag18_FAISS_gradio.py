import os
import gradio as gr
from dotenv import load_dotenv
# langchain에서 지원
from langchain_community.vectorstores import FAISS

# Gradio 챗봇 실습(FAISS)
load_dotenv()
api_key = os.environ['MONOROUTER_API_KEY'].strip()  # strip() : 앞뒤 공백·줄바꿈 제거
base_url = 'https://monogpt.kr/api/monorouter/v1'

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
    dimensions=5,           # text-embedding-3 계열에서만 사용 가능 / 계산은 줄지만 정보도 줄어듦
)

DB_PATH = 'C:/study/_db/Faiss17/'
os.makedirs(DB_PATH, exist_ok=True)

vector_store = FAISS.load_local(
    folder_path=DB_PATH,
    index_name='faiss_index17',
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)

retriever = vector_store.as_retriever(search_kwargs={"k":2})

from langchain_openai import ChatOpenAI
model = ChatOpenAI(
    model='gpt-5.6-luna',
    temperature=0,
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트를 바탕으로 질문에 답변해주세요. 컨텍스트 관련 정보가 없다면,
"주어진 정보로는 답변할 수 없습니다."라고 말씀해 주세요.

컨텍스트: {context}
질문: {input}
답변:
""")

docu_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, docu_chain)

def answer_invoke(message, history):
    response = rag_chain.invoke({"input" : message})
    return response['answer']

demo = gr.ChatInterface(fn=answer_invoke, title="🤖🤖")

demo.launch()