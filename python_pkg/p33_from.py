# import p31_sample
from p31_sample import test

x = 222

def main_func():
    print('x: ', x)
main_func() # x:  222

test()      # x:  111

################################
print("========================")
import p31_sample
p31_sample.test()

# import 하는 단위 > 모듈
# 지금 한 작업은 모듈화. 폴더로 묶어 점으로 내려가게 하면 패키지