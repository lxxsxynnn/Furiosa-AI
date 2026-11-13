from common import get_llm

llm = get_llm(max_tokens=500)

response = llm.invoke("NPU란 무엇인가")

print(response.content)
print("================================")
print(response.usage_metadata)
 # {'input_tokens': 14, 'output_tokens': 1251, 'total_tokens': 1265, 
 #  'input_token_details': {'audio': 0, 'cache_read': 13}, 
 #  'output_token_details': {'audio': 0, 'reasoning': 0}}

#  max_tokens=500 적용 후
# {'input_tokens': 14, 'output_tokens': 500, 'total_tokens': 514, 
#  'input_token_details': {'audio': 0, 'cache_read': 13}, 
#  'output_token_details': {'audio': 0, 'reasoning': 0}}