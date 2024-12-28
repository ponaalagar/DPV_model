from langchain_community.document_loaders import UnstructuredPDFLoader,OnlinePDFLoader
#load documents

from langchain_ollama import OllamaEmbeddings
#to embed text

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma#vector database

import ollama#to use the model

from langchain.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama

from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

doc_path = './data/dpvbook.pdf'
model = 'llama3.2'

if doc_path:
    loader = UnstructuredPDFLoader(file_path=doc_path)
    data = loader.load()
    print('done loading')
else:
    print("something wrong")
    
content = data[0].page_content

text_splitter = RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap=200)
chunks  = text_splitter.split_documents(data)

ollama.pull("nomic-embed-text")

vector_db = Chroma.from_documents(
    documents = chunks,
    embedding = OllamaEmbeddings(model="nomic-embed-text"),
    collection_name ="simple-rag",
)

print("added to vector base")

llm = ChatOllama(model=model)

prompt = PromptTemplate(
    input_variables=["question"],
    template="You are an AI assistant who help students to learn better about data processing and visualization.Answer the following question as best as you can: \n{question}",
)

retriver = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(),llm,prompt=prompt
)

template = """answer the question only on the folowing context{context}
Question:{question}"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriver,"question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

res = chain.invoke(
    input=("tell me something about data processing and visualization")
)
print("-"*30)
print(res)