# LCEL = LangChain Expression Language
# chain = prompt | model | output_parser

# parser : 분석하다, 답변을 다듬다

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv

# 여러 줄 템플릿으로 역할·출력 형식 지정
load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()
base_url = 'https://monogpt.kr/api/monorouter/v1'


template = '''
당신은 영어를 가르치는 10년차 영어 선생님입니다.
주어진 상황에 맞는 영어 회화문을 작성해주세요.
양식은 [FORMAT]을 참고하여 작성해주세요.
현지 Gen-Z들이 많이 쓰는 표현, 슬랭을 포함해 캐주얼하게 작성해주세요.


# 상황
{question}


# FORMAT
영어회화:
한글번역:
문법설명:
'''

prompt = PromptTemplate.from_template(template)

model = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=10,
    api_key=api_key,
    base_url=base_url
)

output_parser = StrOutputParser()

chain = prompt | model | output_parser

input = {'question' : '친구의 생일을 준비하기 위해 모인 사람들'}

response = chain.invoke(input)

print(response)