import os
import time
import logging
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_neo4j import Neo4jGraph
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Neo4jVector

OLLAMA_HOST = "http://192.168.100.3:11434"
OLLAMA_MODEL = "llama3.2:latest"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def setup_connection():
    """Setup connection to Neo4j and return graph and vector retriever"""
    logging.info("Connecting to Neo4j")
    
    # Connect to Neo4j graph
    graph = Neo4jGraph(
        url="bolt://localhost:7687",
        username="neo4j",
        password="admin.admin"
    )
    logging.info("Connected to Neo4j successfully")
    
    # Setup vector retriever from existing graph
    try:
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
        logging.info("Vector index connected successfully")
        vector_retriever = vector_index.as_retriever()
        return graph, vector_retriever
    except Exception as e:
        logging.error(f"Error connecting to vector index: {e}")
        return graph, None

def extract_entities(question):
    """Extract entities from the question using LLM"""
    logging.info(f"Extracting entities from question: {question}")
    
    # Fallback extraction without LLM in case of errors
    words = question.split()
    entities = []
    for i in range(len(words)):
        if len(words[i]) > 0 and words[i][0].isupper() and words[i].lower() not in ["i", "the", "a", "an", "who", "what", "where", "when", "why", "how"]:
            if i < len(words) - 1 and len(words[i+1]) > 0 and words[i+1][0].isupper():
                entities.append(f"{words[i]} {words[i+1]}")
            else:
                entities.append(words[i])
    
    if not entities:
        # If no capitalized entities found, include key nouns
        for word in words:
            if word not in ["is", "am", "are", "was", "were", "be", "being", "been", 
                           "do", "does", "did", "will", "would", "shall", "should",
                           "can", "could", "may", "might", "must", "have", "has", "had",
                           "a", "an", "the", "in", "on", "at", "by", "for", "with", "about",
                           "against", "between", "into", "through", "during", "before",
                           "after", "above", "below", "to", "from", "up", "down", "of"]:
                entities.append(word)
    
    logging.info(f"Extracted entities: {entities}")
    return entities

def querying_neo4j(graph, question):
    """Query Neo4j graph database with extracted entities from the question"""
    logging.info(f"Querying Neo4j with question: {question}")
    
    entities = extract_entities(question)
    
    result = ""
    for entity in entities:
        try:
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
        except Exception as e:
            logging.error(f"Error querying Neo4j for entity {entity}: {e}")
    
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

def get_vector_data(vector_retriever, question):
    """Get relevant vector data from Neo4j"""
    if vector_retriever:
        try:
            vector_results = vector_retriever.invoke(question)
            vector_data = [el.page_content for el in vector_results]
            vector_text = "\n#Document ".join(vector_data) if vector_data else "No relevant documents found."
            return vector_text
        except Exception as e:
            logging.error(f"Error retrieving vector data: {e}")
            return "Error retrieving vector data."
    else:
        return "Vector retriever not available."

def answer_question(graph, vector_retriever, question):
    """Generate answer to question based on Neo4j data"""
    try:
        # Get data from Neo4j graph
        graph_data = querying_neo4j(graph, question)
        logging.info(f"Graph Data retrieved")
        
        # Try to get vector data if available
        try:
            if vector_retriever:
                vector_data = get_vector_data(vector_retriever, question)
                logging.info(f"Vector Data retrieved")
            else:
                vector_data = "Vector retriever not available."
        except Exception as e:
            logging.error(f"Error with vector retrieval: {e}")
            vector_data = "Error retrieving vector data."
        
        context = f"Graph data: {graph_data}\nVector data: {vector_data}"
        
        # Try to use LLaMA for answering
        try:
            template = """Answer the question based only on the following context:
            {context}
            
            Question: {question}
            
            Answer:"""
            
            prompt = ChatPromptTemplate.from_template(template)
            
            llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_HOST, temperature=0)
            chain = prompt | llm | StrOutputParser()
            
            response = chain.invoke({"context": context, "question": question})
            logging.info(f"Final Answer generated with LLM")
            
        except Exception as e:
            logging.error(f"Error generating answer with LLM: {e}")
            response = f"Could not generate LLM answer due to error.\n\nHere's the raw data:\n{context}"
        
        return response
        
    except Exception as e:
        logging.error(f"Error in answer_question: {e}")
        return f"An error occurred while processing your question: {str(e)}"

def main():
    logging.info("Starting Neo4j query interface")
    
    # Setup connection to Neo4j
    try:
        graph, vector_retriever = setup_connection()
        
        print("\nNeo4j Query Interface")
        print("=====================")
        print("Connected to Neo4j database.")
        print("You can now query your existing data.")
        
        while True:
            try:
                user_query = input("\nEnter your question (or 'exit' to quit): ")
                if user_query.lower() == "exit":
                    break
                    
                response = answer_question(graph, vector_retriever, user_query)
                print("\nAnswer:")
                print(response)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logging.error(f"Error processing query: {e}")
                print(f"An error occurred: {e}")
    
    except Exception as e:
        logging.error(f"Failed to set up connections: {e}")
        print(f"Failed to connect to Neo4j: {e}")

if __name__ == "__main__":
    main()