import os
import time
import logging
import re
from fastapi import FastAPI, HTTPException
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

OLLAMA_HOST = "localhost:11434"
OLLAMA_MODEL = "llama3.2:latest"
PDF_PATH = r"C:\Users\SD-LORENZO-PC\pyproject\rndML\fineTuning\com.pdf"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="admin.admin"
)
logging.info("Connected to Neo4j successfully")

def load_pdf(file_path):
    logging.info(f"Loading PDF from: {file_path}")
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        total_pages = len(documents)
        total_content = sum(len(doc.page_content) for doc in documents)
        logging.info(f"Loaded PDF with {total_pages} pages and {total_content} characters")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        split_docs = text_splitter.split_documents(documents)
        logging.info(f"Split into {len(split_docs)} chunks for processing")
        
        return split_docs
    except Exception as e:
        logging.error(f"Error loading PDF: {e}")
        raise

def ingestion(documents):
    logging.info(f"Starting ingestion process for {len(documents)} documents")
    
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_HOST, temperature=0)
    llm_transformer_filtered = LLMGraphTransformer(llm=llm)
    
    batch_size = 5
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        logging.info(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        
        try:
            graph_documents = llm_transformer_filtered.convert_to_graph_documents(batch)
            
            graph.add_graph_documents(
                graph_documents,
                baseEntityLabel=True,
                include_source=True
            )
            logging.info(f"Added batch {i//batch_size + 1} to the graph")
        except Exception as e:
            logging.error(f"Error processing batch {i//batch_size + 1}: {e}")
    
    logging.info("All documents successfully added to the graph")
    
    embed = OllamaEmbeddings(model="mxbai-embed-large", base_url=OLLAMA_HOST)
    
    try:
        vector_index = Neo4jVector.from_existing_graph(
            embedding=embed,
            search_type="hybrid",
            url="bolt://localhost:7687",
            username="neo4j",
            password="admin.admin",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
        logging.info("Vector index created successfully")
        return vector_index.as_retriever()
    except Exception as e:
        logging.error(f"Error creating vector index: {e}")
        raise

def querying_neo4j(question):
    logging.info(f"Querying Neo4j with question: {question}")
    
    prompt = ChatPromptTemplate.from_messages([ 
        ("system", """Extract all person and organization entities from the text.
        Return them as a list like this: ["Entity1", "Entity2", ...].
        Make sure to include only full names of people and organizations."""),
        ("human", "Extract entities from: {question}")
    ])
    
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_HOST, temperature=0)
    
    try:
        response = prompt.invoke({"question": question}) | llm
        response_text = response.content
        
        entities_match = re.search(r'\[(.*?)\]', response_text)
        
        if entities_match:
            entities_str = entities_match.group(1)
            entities = [e.strip().strip('"\'') for e in entities_str.split(',')]
        else:
            words = response_text.split()
            entities = []
            for i in range(len(words)):
                if words[i][0].isupper() and words[i].lower() not in ["i", "the", "a", "an"]:
                    if i < len(words) - 1 and words[i+1][0].isupper():
                        entities.append(f"{words[i]} {words[i+1]}")
                    else:
                        entities.append(words[i])
            
            entities = list(set(entities))
            
        logging.info(f"Extracted entities: {entities}")
        
    except Exception as e:
        logging.error(f"Error extracting entities: {e}")
        entities = []
        
        words = question.split()
        for i in range(len(words)):
            if words[i][0].isupper() and words[i].lower() not in ["i", "the", "a", "an", "who", "what", "where", "when", "why", "how"]:
                if i < len(words) - 1 and words[i+1][0].isupper():
                    entities.append(f"{words[i]} {words[i+1]}")
                else:
                    entities.append(words[i])
    
    result = ""
    for entity in entities:
        query_response = graph.query(
            """MATCH (p)-[r]->(e)
            WHERE p.id = $entity OR p.name = $entity OR p.id CONTAINS $entity OR p.name CONTAINS $entity
            RETURN COALESCE(p.name, p.id) AS source_id, type(r) AS relationship, COALESCE(e.name, e.id) AS target_id
            LIMIT 50""",
            {"entity": entity}
        )
        
        entity_results = [f"{el['source_id'] if el['source_id'] else entity} - {el['relationship']} -> {el['target_id']}" for el in query_response]
        if entity_results:
            result += f"\nRelationships for {entity}:\n"
            result += "\n".join(entity_results) + "\n"
    
    if not result:
        logging.warning(f"No relationships found for entities: {entities}")
        try:
            sample_nodes = graph.query(
                """MATCH (p)-[r]->(e)
                RETURN COALESCE(p.name, p.id) AS source_id, type(r) AS relationship, COALESCE(e.name, e.id) AS target_id
                LIMIT 10"""
            )
            if sample_nodes:
                result = "No exact matches found, but here are some available entities in the graph:\n"
                result += "\n".join([f"{el['source_id']} - {el['relationship']} -> {el['target_id']}" for el in sample_nodes])
        except Exception as e:
            logging.error(f"Error getting sample nodes: {e}")
    
    return result

def full_retriever(question: str, vector_retriever):
    graph_data = querying_neo4j(question)
    logging.info(f"Graph Data: {graph_data}")
    
    vector_data = [el.page_content for el in vector_retriever.invoke(question)]
    vector_text = "\n#Document ".join(vector_data) if vector_data else "No relevant documents found."
    
    return f"Graph data: {graph_data}\nVector data: {vector_text}"

def querying_ollama(question, vector_retriever):
    logging.info(f"Querying LLaMA with question: {question}")
    
    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_HOST, temperature=0)
    chain = prompt | llm | StrOutputParser()
    
    context = full_retriever(question, vector_retriever)
    response = chain.invoke({"context": context, "question": question})
    logging.info(f"Final Answer: {response}")
    return response

def main():
    logging.info("Starting PDF to Neo4j RAG pipeline")
    
    documents = load_pdf(PDF_PATH)
    
    vector_retriever = ingestion(documents)
    
    while True:
        try:
            user_query = input("\nEnter your question (or 'exit' to quit): ")
            if user_query.lower() == "exit":
                break
                
            response = querying_ollama(user_query, vector_retriever)
            print("\nAnswer:")
            print(response)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()