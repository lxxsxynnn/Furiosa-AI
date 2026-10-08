# pip install pypdf
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# pdf 문서 불러오기
path = 'C:/study/_data/'
pdf_loader = PyPDFLoader(path + 'attention_is_all_you_need.pdf')
pdf_docs = pdf_loader.load()

print(type(pdf_docs))   # <class 'list'>
print(len(pdf_docs))    # 15
# print(pdf_docs)
'''
[
    Document(
        metadata={
            'producer': 'pdfTeX-1.40.25',
            'creator': 'LaTeX with hyperref',
            'creationdate': '2024-04-10T21:11:43+00:00',
            'author': '',
            'keywords': '',
            'moddate': '2024-04-10T21:11:43+00:00',
            'ptex.fullbanner': 'This is pdfTeX, Version 3.141592653-2.6-1.40.25 (TeX Live 2023) kpathsea version 6.3.5',
            'subject': '',
            'title': '',
            'trapped': '/False',
            'source': 'C:/study/_data/attention_is_all_you_need.pdf',
            'total_pages': 15,
            'page': 0,
            'page_label': '1',
        },
        page_content='Provided proper attribution is provided, Google hereby grants ... arXiv:1706.03762v7  [cs.CL]  2 Aug 2023',
    ),
    Document(
        metadata={
            'producer': 'pdfTeX-1.40.25',
            ...
            'page': 1,
            'page_label': '2',
        },
        page_content='1 Introduction\nRecurrent neural networks, long short-term memory [13] ... when generating the next.\n2',
    ),
    ...
    Document(
        metadata={
            'producer': 'pdfTeX-1.40.25',
            ...
            'page': 14,
            'page_label': '15',
        },
        page_content='Input-Input Layer5\nThe\nLaw\nwill\nnever\nbe\nperfect ... learned to perform different tasks.\n15',
    ),
]                       # 15개
'''

load_dotenv()
api_key = os.environ['MONOROUTER_API_KEY'].strip()
base_url = 'https://monogpt.kr/api/monorouter/v1'

# 분할 기준 만들기
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""],    # 기본값과 같음 (생략 가능)
)

# 문서 분할
split_pdf = text_splitter.split_documents(pdf_docs)   # 위에서 읽은 pdf_docs를 나눔. load_and_split은 PDF를 한 번 더 읽음
print(len(split_pdf))   # 103

# 임베딩
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
    # dimensions=5,         # 기본 1536차원으로 저장. rag19_PyPDF_2_Chatbot도 같은 차원으로 불러야 함
)


DB_PATH = 'C:/study/_db/Faiss19/'
os.makedirs(DB_PATH, exist_ok=True)

db = FAISS.from_documents(
    documents=split_pdf,
    embedding=embeddings,
)

db.save_local(
    folder_path=DB_PATH,
    index_name='faiss_index19',
)