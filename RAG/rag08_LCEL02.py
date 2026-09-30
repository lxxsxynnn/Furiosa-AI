# LCEL = LangChain Expression Language
# chain = prompt | model | output_parser

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

# 프롬프트와 모델 연결하기 - 파라미터가 2개인 프롬프트
load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()  # strip() : 줄바꿈, 띄어쓰기 무시
base_url = 'https://monogpt.kr/api/monorouter/v1'

prompt = PromptTemplate.from_template("{topic}에 대해 알기 쉽게 {how} 설명해주세요.")

model = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=10,
    api_key=api_key,
    base_url=base_url
)

chain = prompt | model      # 두 가지를 연결해서 작업하겠다는 의미

input = {'topic' : '양자컴퓨터 학습 원리',
         'how':'간단한 문장으로 요약해서',}

response = chain.invoke(input)

print(response.content)