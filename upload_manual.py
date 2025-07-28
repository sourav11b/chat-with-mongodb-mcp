import os
import time
from dotenv import load_dotenv
from pymongo import MongoClient
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.operations import SearchIndexModel

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_voyageai import VoyageAIEmbeddings


load_dotenv()

ATLAS_URI = os.getenv("ATLAS_URI")
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME")
ATLAS_DB_NAME = os.getenv("ATLAS_DB_NAME")
ATLAS_COLLECTION_NAME_MANUALS = os.getenv("ATLAS_COLLECTION_NAME_MANUALS")

client = MongoClient(ATLAS_URI)

# Access your database and collection
database = client[ATLAS_DB_NAME]
collection = database[ATLAS_COLLECTION_NAME_MANUALS]

def create_vector_index_definition():
    print("starting to create vector index")

    # Check if the collection exists, if not, create it
    if ATLAS_COLLECTION_NAME_MANUALS not in database.list_collection_names():
        print(f"Collection '{ATLAS_COLLECTION_NAME_MANUALS}' does not exist. Creating it...")
        # Explicitly create the collection.
        # This ensures the collection exists before attempting to create an index on it.
        database.create_collection(ATLAS_COLLECTION_NAME_MANUALS)
        print(f"Collection '{ATLAS_COLLECTION_NAME_MANUALS}' created.")
    else:
        print(f"Collection '{ATLAS_COLLECTION_NAME_MANUALS}' already exists.")
    search_index_model = SearchIndexModel(
      definition={
        "fields": [
          {
            "type": "vector",
            "numDimensions": 512,
            "path": "embedding",
            "similarity": "cosine"
          }
        ]
      },
      name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
      type="vectorSearch"
    )

    result = collection.create_search_index(model=search_index_model)
    print("New search index named " + result + " is building.")

    # Wait for initial sync to complete
    print("Polling to check if the index is ready. This may take up to a minute.")
    predicate=None
    if predicate is None:
      predicate = lambda index: index.get("queryable") is True

    while True:
      indices = list(collection.list_search_indexes(result))
      if len(indices) and predicate(indices[0]):
        break
      time.sleep(5)
    print(result + " is ready for querying.")    
    
def load_manuals():
    
    print("starting to chunk and upload documents")
    loader = PyPDFLoader("content/manual.pdf")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    docs = text_splitter.split_documents(data)
    # insert the documents in MongoDB Atlas Vector Search
    x = MongoDBAtlasVectorSearch.from_documents(
         documents=docs, embedding=VoyageAIEmbeddings( model="voyage-3-lite"), collection=collection, index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
    )
     
    print("documents ready to query")
     
     
def main():
    print("---STARTING PROCESS---")
    create_vector_index_definition()
    load_manuals()
    print("---DONE---")
if __name__ == "__main__":
    main()