import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# monogpt api 키 연결
load_dotenv()
api_key = os.environ['MONOROUTER_API_KEY'].strip()  # strip() : 줄바꿈, 띄어쓰기 무시
base_url = 'https://monogpt.kr/api/monorouter/v1'

llm = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=2,
    api_key=api_key,
    base_url=base_url
)

response = llm.invoke('gohan o tabemashitaka?')
print(response.text) # It means: **“Did you eat?”** or literally **“Did you eat a meal/rice?”**
print(response.content)
print(response)
'''
content='「ご飯を食べましたか？」  \n(*Gohan o tabemashita ka?*)\n\nIt means: **“Did you eat?”** / **“Have you eaten a meal?”**\n\n- ご飯 (gohan) = meal/rice  \n- を (o) = object marker  \n- 食べました (tabemashita) = ate  \n- か (ka) = question marker'
additional_kwargs={'refusal': None}
response_metadata={
    'token_usage': {
        'completion_tokens': 87,
        'prompt_tokens': 14,
        'total_tokens': 101,
        'completion_tokens_details': {
            'accepted_prediction_tokens': 0,
            'audio_tokens': 0,
            'reasoning_tokens': 0,
            'rejected_prediction_tokens': 0,
            'text_tokens': None,
        },
        'prompt_tokens_details': {
            'audio_tokens': 0,
            'cache_write_tokens': 0,
            'cached_tokens': 0,
            'image_tokens': None,
            'text_tokens': None,
        },
    },
    'model_provider': 'openai',
    'model_name': 'gpt-5.6-terra',
    'system_fingerprint': None,
    'id': 'chatcmpl-ETg4vNcuUA3yh0VHr1gYyC4bavsuP',
    'service_tier': 'default',
    'finish_reason': 'stop',
    'logprobs': None,
}
id='lc_run--01a0f089-f445-7703-aeb4-6f5bd2dbdf70-0'
tool_calls=[]
invalid_tool_calls=[]
usage_metadata={
    'input_tokens': 14,
    'output_tokens': 87,
    'total_tokens': 101,
    'input_token_details': {
        'audio': 0,
        'cache_read': 0,
        'cache_creation': 0,
    },
    'output_token_details': {
        'audio': 0,
        'reasoning': 0,
    },
}
'''