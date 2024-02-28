import os
from langchain_community.document_loaders import GoogleDriveLoader
from langchain_googledrive.retrievers import GoogleDriveRetriever
from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI  # Import the OpenAI language model
from langchain_community.document_loaders import GoogleDriveLoader
from pathlib import Path


# Set the path to the credentials.json file
os.environ['GOOGLE_ACCOUNT_FILE'] = '/workspaces/Sep10/credentials.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/workspaces/Sep10/credentials.json'

# Set the path to the token.json file
os.environ['GOOGLE_TOKEN_PATH'] = '/workspaces/Sep10/token.json'

# Initialize the GoogleDriveLoader with the specified paths
loader = GoogleDriveLoader(
    folder_id="1bM-uILBTV_QWmIQ_ohghyrOHfXiZO9zT",  # Replace with your actual folder ID
    credentials_path=Path('/workspaces/Sep10/credentials.json'),
    token_path=Path('/workspaces/Sep10/token.json'),
    recursive=False  # Optional: configure whether to recursively fetch files from subfolders. Defaults to False.
)


# Initialize the Google Drive Retriever
retriever = GoogleDriveRetriever(
    folder_id="1bM-uILBTV_QWmIQ_ohghyrOHfXiZO9zT",
    num_results=2  
)

# Initialize the OpenAI language model
openai_model = ChatOpenAI(model="gpt-3.5-turbo")  # Replace with your specific OpenAI model

# Initialize the agent with the Google Drive retriever and the OpenAI language model
agent = initialize_agent(
    tools=[retriever, loader],  
    llm=openai_model,  
    agent=AgentType.STRUCTURED_CHAT,  # Choose the appropriate agent type (e.g., STRUCTURED_CHAT)
)

# Use the agent to interact and retrieve documents from Google Drive
result = agent.run("Search in google drive, documents about 'machine learning'")
print(result)