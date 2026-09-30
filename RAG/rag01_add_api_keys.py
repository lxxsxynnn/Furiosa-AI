import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# api키 직접 넣기
load_dotenv()
# openai_api_key = ''
openai_api_key = os.getenv('RAG01_API_KEY')

llm = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    openai_api_key=openai_api_key,
)

response = llm.invoke('안녕하세요.')
print(response.content) # 안녕하세요! 무엇을 도와드릴까요?
# print(response)     # 가독성이 떨어져서 답변만 보려면 response.content로 출력
'''
content='안녕하세요! 무엇을 도와드릴까요?'
additional_kwargs={'refusal': None}
response_metadata={
    'token_usage': {
        'completion_tokens': 14, 'prompt_tokens': 9, 'total_tokens': 23,
        'completion_tokens_details': {
            'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0, 'text_tokens': None
        },
        'prompt_tokens_details': {
            'audio_tokens': 0, 'cache_write_tokens': 0, 'cached_tokens': 0, 'image_tokens': None, 'text_tokens': None
        }
    },
    'model_provider': 'openai',
    'model_name': 'gpt-5.6-terra',
    'system_fingerprint': None,
    'id': 'chatcmpl-ETdNlbFFBAeZAFTqlcZhUjT3Z8hxY',
    'service_tier': 'default',
    'finish_reason': 'stop',
    'logprobs': None
}
id='lc_run--01a0efeb-cf9f-7553-b71b-c2372e7d27a1-0'
tool_calls=[]
invalid_tool_calls=[]
usage_metadata={
    'input_tokens': 9, 'output_tokens': 14, 'total_tokens': 23,
    'input_token_details': {
        'audio': 0, 'cache_read': 0, 'cache_creation': 0
    },
    'output_token_details': {'audio': 0, 'reasoning': 0}
}
'''