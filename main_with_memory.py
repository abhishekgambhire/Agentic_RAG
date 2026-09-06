import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Modern standalone Chroma library (removes the LangChainDeprecationWarning)
from langchain_chroma import Chroma 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver 
from langchain_core.tools import create_retriever_tool
from langchain.agents import create_agent 
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "./chroma_db_storage"
DOCUMENT_PATH = "The Big Bang.pdf"

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# FIX: Ingest data if the directory doesn't exist OR if it is completely empty
if not os.path.exists(CHROMA_PATH) or not os.listdir(CHROMA_PATH):
    if os.path.exists(DOCUMENT_PATH):
        print(f"Ingesting '{DOCUMENT_PATH}' into database...")
        loader = PyPDFLoader(DOCUMENT_PATH)
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(loader.load())
        vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_PATH)
    else:
        raise FileNotFoundError(f"Could not find the document at: {DOCUMENT_PATH}")
else:
    print("Loading existing database directory...")
    vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# 2. Package the retriever into a tool
retriever = vector_store.as_retriever(search_kwargs={"k": 4})
retriever_tool = create_retriever_tool(
    retriever,
    name="document_search",
    description="Search for information within the uploaded documents. You must look up details using this tool for all user queries."
)

# 3. Create the Agent with Memory Saver and System Prompt
agent = create_agent(
    model=llm,
    tools=[retriever_tool], 
    system_prompt="""You are a helpful assistant specialized in answering questions about the uploaded documents. 
    CRITICAL: For every question the user asks, you MUST always call the 'document_search' tool to check the contents of the document first. 
    Do not answer from your pre-trained knowledge base without checking the document tool context first.""",
    checkpointer=InMemorySaver()
)

thread_config = {"configurable": {"thread_id": "rag_session_1"}}

# 4. Interactive While Loop
if __name__ == "__main__":
    print("\n--- RAG Agent Chat Initialized ---")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    
    while True:
        query = input("User: ")
        
        if query.strip().lower() in ["exit", "quit"]:
            print("Exiting chat. Goodbye!")
            break
            
        if not query.strip():
            continue
            
        try:
            response = agent.invoke({"messages": [("user", query)]}, thread_config)
            
            # Extract the raw message content
            agent_msg = response['messages'][-1].content
            
            # Format output in case Gemini returns structured content blocks 
            if isinstance(agent_msg, list):
                text_content = " ".join([block['text'] for block in agent_msg if block.get('type') == 'text'])
                print(f"Agent: {text_content}\n")
            else:
                print(f"Agent: {agent_msg}\n")
            
        except Exception as e:
            print(f"An error occurred: {e}\n")