import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# os.environ에 api키 등록
load_dotenv()
openai_api_key = os.getenv('RAG01_API_KEY')

# 컴퓨터의 환경변수에 변수 추가
# 휘발성이기 때문에 저장되지 않음
os.environ['OPENAI_API_KEY'] = openai_api_key

llm = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    # openai_api_key=openai_api_key,
)

response = llm.invoke('내가 누구게?')
# 아직 메모리에 정보가 저장되지 않음
print(response.content) # 아직은 알 수 없어요! 힌트를 주시면 맞혀볼게요 😄