# 데코레이터: 함수를 불러오는 함수
def hello():
    print("안녕하세요")

hello()

print("================ 1 ================")
def deco(func):
    def wrapper():
        print("함수 실행 전")
        func()
        print("함수 실행 후")
    return wrapper

@deco
def hello2():
    print("안녕하세요2")

hello2()
'''
함수 실행 전
안녕하세요2
함수 실행 후
'''