# TypedDict : dict의 키마다 값 타입 정하기 (실행할 때 타입을 검사하지는 않음)
from typing import Dict, TypedDict

sample_dict : Dict[str, str] = {
    'name' : 'Jack Song',
    'age' : 26,
    # 'age' : '26',
    # 'job' : 'Artist',
    'job' : 100,
}

# print(sample_dict)  # {'name': 'Jack Song', 'age': '26', 'job': 'Artist'}
# Type의 강제성은 없음
# {'name': 'Jack Song', 'age': 26, 'job': 'Artist'}
# {'name': 'Jack Song', 'age': 26, 'job': 100}

########################################################################
class Person(TypedDict):
    name : str
    age : int
    job : str

person = Person(name='Jack Song', age=26, job='Artist')
print(person)   # {'name': 'Jack Song', 'age': 26, 'job': 'Artist'}
########################################################################

person2: Person = {'name': 'Jimmy Lee', 'age': 24, 'job': 'singer'}
print(person2)  # {'name': 'Jimmy Lee', 'age': 24, 'job': 'singer'}