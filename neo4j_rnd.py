import os
import time
import logging
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

# Configuration
OLLAMA_HOST = "http://192.168.100.3:11434"
OLLAMA_MODEL = "llama3.2:latest" 

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Neo4j connection
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="admin.admin"
)
logging.info("Connected to Neo4j successfully")

def ingestion(text):
    logging.info("Starting ingestion process")
    
    documents = [Document(page_content=text)]
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_HOST, temperature=0)
    llm_transformer_filtered = LLMGraphTransformer(llm=llm)
    
    graph_documents = llm_transformer_filtered.convert_to_graph_documents(documents)
    
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )
    logging.info("Documents successfully added to the graph")
    
    embed = OllamaEmbeddings(model="mxbai-embed-large", base_url=OLLAMA_HOST)
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

def querying_neo4j(question):
    logging.info("Querying Neo4j with question: %s", question)
    
    class Entities(BaseModel):
        names: list[str] = Field(..., description="All entities from the text")
    
    # Extract entities with a simple approach
    prompt = ChatPromptTemplate.from_messages([ 
        ("system", """Extract all person and organization entities from the text.
        Return them as a list like this: ["Entity1", "Entity2", ...].
        Make sure to include only full names of people and organizations."""),
        ("human", "Extract entities from: {question}")
    ])
    
    # Use regular ChatOllama and parse the response manually
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_HOST, temperature=0)
    
    try:
        # Get raw response
        response = prompt.invoke({"question": question}) | llm
        response_text = response.content
        
        # Basic parsing - find entities in the response
        # Look for lists in the format ["Entity1", "Entity2"]
        import re
        entities_match = re.search(r'\[(.*?)\]', response_text)
        
        if entities_match:
            # Extract the content inside brackets and split by commas
            entities_str = entities_match.group(1)
            # Split by comma and clean up quotes and spaces
            entities = [e.strip().strip('"\'') for e in entities_str.split(',')]
        else:
            # Fallback: try to find names by looking for capitalized words
            words = response_text.split()
            entities = []
            for i in range(len(words)):
                if words[i][0].isupper() and words[i].lower() not in ["i", "the", "a", "an"]:
                    if i < len(words) - 1 and words[i+1][0].isupper():
                        entities.append(f"{words[i]} {words[i+1]}")
                    else:
                        entities.append(words[i])
            
            # Remove duplicates and filter out common words
            entities = list(set(entities))
            
        logging.info("Extracted entities: %s", entities)
        
    except Exception as e:
        logging.error(f"Error extracting entities: {e}")
        entities = []
        
        # Try to extract entities from the question as fallback
        words = question.split()
        for i in range(len(words)):
            if words[i][0].isupper() and words[i].lower() not in ["i", "the", "a", "an", "who", "what", "where", "when", "why", "how"]:
                if i < len(words) - 1 and words[i+1][0].isupper():
                    entities.append(f"{words[i]} {words[i+1]}")
                else:
                    entities.append(words[i])
    
    # Query the Neo4j database for each entity
    result = ""
    for entity in entities:
        query_response = graph.query(
            """MATCH (p)-[r]->(e)
            WHERE p.id = $entity OR p.name = $entity
            RETURN p.id AS source_id, type(r) AS relationship, e.id AS target_id
            LIMIT 50""",
            {"entity": entity}
        )
        
        entity_results = [f"{el['source_id'] if el['source_id'] else entity} - {el['relationship']} -> {el['target_id']}" for el in query_response]
        if entity_results:
            result += "\n".join(entity_results) + "\n"
    
    return result

def full_retriever(question: str):
    graph_data = querying_neo4j(question)
    logging.info("Graph Data: %s", graph_data)
    
    vector_data = [el.page_content for el in vector_retriever.invoke(question)]
    
    return f"Graph data: {graph_data}\nVector data: {'#Document '.join(vector_data)}"

def querying_ollama(question, vector_retriever):
    logging.info("Querying LLaMA with question: %s", question)
    
    template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_HOST, temperature=0)
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({"context": full_retriever(question), "question": question})
    logging.info("Final Answer: %s", response)
    return response

# Test ingestion and querying
if __name__ == "__main__":
    sample_text = """
    Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
    She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
    Pierre Curie was a French physicist, a pioneer in crystallography, magnetism, piezoelectricity, and radioactivity.
    In 1903, Marie and Pierre Curie were awarded the Nobel Prize in Physics along with Henri Becquerel for their work on radiation.
    """
    
    vector_retriever = ingestion(sample_text)
    response = querying_ollama("Who are Marie Curie and Pierre Curie?", vector_retriever)
    print(response)