import os
import gradio as gr
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# 프롬프트 규칙으로 일상 대화와 문서 질문 나눠 답하기
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

# 저장된 데이터 불러오기
DB_PATH = 'C:/study/_db/Faiss20/'
os.makedirs(DB_PATH, exist_ok=True)

vector_store = FAISS.load_local(
    folder_path=DB_PATH,
    index_name='faiss_index20',
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)

# 검색기 생성
# retriever = vector_store.as_retriever(search_kwargs={"k":2})
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 2,
        "score_threshold": 0.7,     # 유사도 0.7 미만은 검색에서 제외
    }
)

# 유사도가 낮으면 아래 메시지가 나옴
# No relevant docs were retrieved using the relevance score threshold 0.7

# 모델 생성
model = ChatOpenAI(
    model='gpt-5.6-luna',
    temperature=0,
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)

from langchain_core.prompts import MessagesPlaceholder  # 프롬프트에 history를 넣을 공간 만들기
from langchain_core.messages import HumanMessage, AIMessage

# 프롬프트 생성
prompt = ChatPromptTemplate.from_messages([
    ("system", """
        당신은 강력한 AI 챗봇입니다.
        다음 규칙에 따라 답변하세요.

        1. 이전 대화에서 확인할 수 있는 정보는 이전 대화를 참고해 답변하세요.
        2. 일반적인 인사나 일상 대화는 자연스럽게 답변하세요.
        3. 문서에 관한 질문은 제공된 컨텍스트를 근거로 답변하세요.
        4. 문서에 관한 질문인데 컨텍스트에 관련 정보가 없다면
        "주어진 정보로는 답변할 수 없습니다."라고 답변하세요.

        컨텍스트: {context}
    """),
    MessagesPlaceholder(variable_name='chat_history'),
    ("human", "{input}"),
]) 

# 체인 생성
docu_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, docu_chain)

# 답변 생성 함수 정의
# ChatInterface가 (입력한 메시지, 이전 대화 리스트)로 부름. history는 프롬프트에만 들어가고, 검색은 이번 질문(input)으로만 함
def answer_invoke(message, history):
    chat_history=[]

    for item in history[-10:]:      # 최근 10개 메시지만 전달
        content = item["content"]

        if item["role"] == "user":
            chat_history.append(HumanMessage(content=content))
        elif item["role"]=="assistant":
            chat_history.append(AIMessage(content=content))
        
    response = rag_chain.invoke({
        "input" : message,
        "chat_history" : chat_history
        })

    # 터미널 상으로 실제 검색된 문서 확인 (기준값을 넘은 것만, 최대 2개)
    for i, doc in enumerate(response['context'], start=1):
        print(f"검색 문서 {i}")
        print("출처: ", doc.metadata.get("source"))
        print("내용: ", doc.page_content)
        print("=========================================")
    
    return response['answer']

# 챗봇 인터페이스 정의
demo = gr.ChatInterface(fn=answer_invoke, title="YOLO")

# 챗봇 실행
demo.launch()
