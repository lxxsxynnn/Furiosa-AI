import os

# 환경변수에 따른 키 확인하기
# key = os.getenv('OPENAI_API_KEY')   # 사용자 환경변수에 추가한 값
key = os.getenv('RAG01_API_KEY')    # .env에 추가한 값(vs code에서 자동으로 인식을 해주는 것 뿐이고, 원칙상으로는 dotenv를 통해 불러와야 함)

if key is None:
    print('OPENAI_API_KEY is not found')                # 시스템 환경변수 삭제 후에 출력됨
else:
    print("key length: ", len(key))                     # key length:  164
    print("key check: ", key[:8] + "..." + key[-4:])    # key check:  sk-proj-...KecA