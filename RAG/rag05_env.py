import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# .env 파일에서 api 키 불러오기
load_dotenv()
openai_api_key = os.getenv('GEMINI_API_KEY')

# llm = ChatOpenAI(
#     model_name='gpt-5.6-terra',
#     temperature=10,
#     openai_api_key=openai_api_key,
# )

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=2.0,
)

response = llm.invoke('내가 누구게?')
print(response.text) # 아직은 알 수 없어요! 힌트를 주시면 맞혀볼게요 😄