import os
import time
import logging
import re
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configuration
OLLAMA_HOST = "http://192.168.100.3:11434"
PDF_PATH = r"C:\Users\SD-LORENZO-PC\pyproject\rndML\fineTuning\rnd\com.pdf"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize FastAPI app
app = FastAPI(title="Neo4j RAG Chat API", 
              description="API for chatting with documents stored in Neo4j using RAG",
              version="1.0.0")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Neo4j connection
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="admin.admin"
)

# Global variable to store the vector retriever
vector_retriever = None
initialization_complete = False
initialization_error = None

# Pydantic models for API requests and responses
class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    session_data: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = None
    conversation_id: Optional[str] = None
    processing_time: float

class PdfRequest(BaseModel):
    pdf_path: str

class StatusResponse(BaseModel):
    status: str
    message: str
    initialized: bool

def load_pdf(file_path):
    """Load and split a PDF document."""
    logging.info(f"Loading PDF from: {file_path}")
    try:
        # Load the PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Get total pages and content size for logging
        total_pages = len(documents)
        total_content = sum(len(doc.page_content) for doc in documents)
        logging.info(f"Loaded PDF with {total_pages} pages and {total_content} characters")
        
        # Split documents into chunks for better processing
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
    """Process documents and ingest into Neo4j and vector store."""
    logging.info(f"Starting ingestion process for {len(documents)} documents")
    
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_HOST, temperature=0)
    llm_transformer_filtered = LLMGraphTransformer(llm=llm)
    
    # Process in batches to avoid overloading the LLM
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
    
    # Create vector embeddings
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
    """Extract entities from question and query Neo4j graph database."""
    logging.info(f"Querying Neo4j with question: {question}")
    
    # Extract entities with a simple approach
    prompt = ChatPromptTemplate.from_messages([ 
        ("system", """Extract all person and organization entities from the text.
        Return them as a list like this: ["Entity1", "Entity2", ...].
        Make sure to include only full names of people and organizations."""),
        ("human", "Extract entities from: {question}")
    ])
    
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_HOST, temperature=0)
    
    try:
        # Get raw response
        response = prompt.invoke({"question": question}) | llm
        response_text = response.content
        
        # Basic parsing - find entities in the response
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
            
        logging.info(f"Extracted entities: {entities}")
        
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
    found_entities = []
    
    for entity in entities:
        # Try both exact and fuzzy matches
        query_response = graph.query(
            """MATCH (p)-[r]->(e)
            WHERE p.id = $entity OR p.name = $entity OR p.id CONTAINS $entity OR p.name CONTAINS $entity
            RETURN COALESCE(p.name, p.id) AS source_id, type(r) AS relationship, COALESCE(e.name, e.id) AS target_id
            LIMIT 50""",
            {"entity": entity}
        )
        
        entity_results = [f"{el['source_id'] if el['source_id'] else entity} - {el['relationship']} -> {el['target_id']}" for el in query_response]
        if entity_results:
            found_entities.append(entity)
            result += f"\nRelationships for {entity}:\n"
            result += "\n".join(entity_results) + "\n"
    
    if not result:
        logging.warning(f"No relationships found for entities: {entities}")
        # Get some sample nodes as fallback
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
    
    return result, found_entities

def full_retriever(question: str, retriever):
    """Combine graph and vector results for context."""
    graph_data, found_entities = querying_neo4j(question)
    logging.info(f"Graph Data: {graph_data}")
    
    # Get vector results
    vector_results = retriever.invoke(question)
    vector_data = [el.page_content for el in vector_results]
    vector_text = "\n#Document ".join(vector_data) if vector_data else "No relevant documents found."
    
    sources = found_entities + [f"Document chunk {i+1}" for i in range(len(vector_data))]
    
    return f"Graph data: {graph_data}\nVector data: {vector_text}", sources

def chat_with_rag(question, retriever):
    """Generate answer using combined context."""
    logging.info(f"Processing chat question: {question}")
    start_time = time.time()
    
    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_HOST, temperature=0)
    chain = prompt | llm | StrOutputParser()
    
    context, sources = full_retriever(question, retriever)
    response = chain.invoke({"context": context, "question": question})
    
    processing_time = time.time() - start_time
    
    return {
        "answer": response,
        "sources": sources,
        "processing_time": round(processing_time, 2)
    }

async def initialize_system():
    """Initialize the system by loading PDF and creating vector store."""
    global vector_retriever, initialization_complete, initialization_error
    
    try:
        logging.info("Starting system initialization")
        documents = load_pdf(PDF_PATH)
        vector_retriever = ingestion(documents)
        initialization_complete = True
        logging.info("System initialization complete")
    except Exception as e:
        initialization_error = str(e)
        logging.error(f"System initialization failed: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup."""
    # Start initialization in background to allow API to start faster
    background_tasks = BackgroundTasks()
    background_tasks.add_task(initialize_system)
    await initialize_system()

@app.get("/", response_model=StatusResponse)
async def root():
    """Root endpoint to check API status."""
    if initialization_complete:
        return StatusResponse(status="ok", message="Neo4j RAG Chat API is ready", initialized=True)
    elif initialization_error:
        return StatusResponse(status="error", message=f"Initialization failed: {initialization_error}", initialized=False)
    else:
        return StatusResponse(status="initializing", message="System is initializing, please wait", initialized=False)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint to ask questions and get answers."""
    global vector_retriever, initialization_complete
    
    if not initialization_complete:
        if initialization_error:
            raise HTTPException(status_code=500, detail=f"System initialization failed: {initialization_error}")
        else:
            raise HTTPException(status_code=503, detail="System is still initializing, please try again later")
    
    try:
        result = chat_with_rag(request.question, vector_retriever)
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            conversation_id=request.conversation_id,
            processing_time=result["processing_time"]
        )
    except Exception as e:
        logging.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-pdf", response_model=StatusResponse)
async def load_new_pdf(pdf_request: PdfRequest, background_tasks: BackgroundTasks):
    """Load a new PDF file and update the knowledge base."""
    global PDF_PATH, initialization_complete, initialization_error
    
    try:
        # Update the PDF path
        PDF_PATH = pdf_request.pdf_path
        # Reset initialization state
        initialization_complete = False
        initialization_error = None
        
        # Start reinitialization in background
        background_tasks.add_task(initialize_system)
        
        return StatusResponse(
            status="initializing", 
            message=f"Started loading new PDF from {PDF_PATH}", 
            initialized=False
        )
    except Exception as e:
        logging.error(f"Error loading new PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/direct-chat")
async def direct_chat(request: ChatRequest):
    """Chat directly with the model without RAG for comparison."""
    try:
        start_time = time.time()
        
        llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_HOST, temperature=0)
        response = llm.invoke(request.question)
        
        processing_time = time.time() - start_time
        
        return ChatResponse(
            answer=response.content,
            sources=None,
            conversation_id=request.conversation_id,
            processing_time=round(processing_time, 2)
        )
    except Exception as e:
        logging.error(f"Error in direct chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)