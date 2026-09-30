# LCEL = LangChain Expression Language
# chain = prompt | model | output_parser

# parser : 분석하다, 답변을 다듬다

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv

# 출력 파서까지 체인에 연결해보기
load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()
base_url = 'https://monogpt.kr/api/monorouter/v1'

prompt = PromptTemplate.from_template("{topic}에 대해 알기 쉽게 설명해주세요.")

model = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=10,
    api_key=api_key,
    base_url=base_url
)

output_parser = StrOutputParser()

chain = prompt | model | output_parser

input = {'topic' : '양자컴퓨터 학습 원리'}

response = chain.invoke(input)

print(response)     # output_parser가 알아서 content 내용만 string 형태로 뽑아냄