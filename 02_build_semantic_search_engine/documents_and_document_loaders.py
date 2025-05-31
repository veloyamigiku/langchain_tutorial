import argparse
import langchain_chroma.vectorstores
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.runnables import chain
import langchain_chroma
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from typing import List

"""
wget https://www.sqlite.org/2025/sqlite-autoconf-3490200.tar.gz
tar zxvf sqlite-autoconf-3490200.tar.gz 
cd sqlite-autoconf-3490200
./configure 
make
make install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
"""

def main(args):
  
  # Loading documents
  file_path = 'nke-10k-2023.pdf'
  loader = PyPDFLoader(file_path=file_path)

  docs = loader.load()
  print(len(docs))
  print(f'{docs[0].page_content[:200]}\n')
  print(docs[0].metadata)

  # Splitting
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True)
  all_splits = text_splitter.split_documents(docs)
  print(len(all_splits))

  # Embeddings
  embeddings = OllamaEmbeddings(model='llama3.1')

  vector1 = embeddings.embed_query(all_splits[0].page_content)
  vector2 = embeddings.embed_query(all_splits[1].page_content)

  assert len(vector1) == len(vector2)
  print(f'Generated vectors of length {len(vector1)}\n')
  print(vector1[:10])

  # Vector stores
  vector_store = Chroma(
    collection_name='example_collection',
    embedding_function=embeddings,
    persist_directory='./chroma_langchain_db')
  """
  ids = []
  with tqdm(total=len(all_splits)) as pbar:
    for idx in range(0, len(all_splits), 10):
      if len(all_splits) - idx < 10:
        pbar.update(len(all_splits) - idx)
      else:
        pbar.update(10)
      ids.append(vector_store.add_documents(documents=all_splits[idx:idx + 10]))
  ids = vector_store.add_documents(documents=all_splits)
  print(ids)
  """
  results = vector_store.similarity_search(
    'How many distribution centers does Nike have in the US?'
  )
  print(results[0])

  results = vector_store.similarity_search_with_score('What was Nike\'s revenue in 2023?')
  doc, score = results[0]
  print(f'Score: {score}\n')
  print(doc)

  embedding = embeddings.embed_query('How were Nike\'s margins impacted in 2023?')
  results = vector_store.similarity_search_by_vector(embedding)
  print(results[0])

  # Retrievers
  retriever = vector_store.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 1}
  )
  retriever_res = retriever.batch([
    'How many distribution centers does Nike have in the US?',
    'When was Nike incorporated?',
  ])
  print(retriever_res)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  args = parser.parse_args()
  main(args=args)
