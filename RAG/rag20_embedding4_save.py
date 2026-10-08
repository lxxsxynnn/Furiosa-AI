# pip install pypdf
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# pdf 문서 불러오기
path = 'C:/study/_data/'
pdf_loader = PyPDFLoader(path + 'attention_is_all_you_need.pdf')
pdf_docs = pdf_loader.load()

print(type(pdf_docs))   # <class 'list'>
print(len(pdf_docs))    # 15

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
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={
        "device" : "cpu",
        # "local_files_only" : True,
    },
)



DB_PATH = 'C:/study/_db/Faiss20/'
os.makedirs(DB_PATH, exist_ok=True)

db = FAISS.from_documents(
    documents=split_pdf,
    embedding=embeddings,
)

db.save_local(
    folder_path=DB_PATH,
    index_name='faiss_index20',
)