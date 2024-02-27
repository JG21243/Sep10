from langchain_googledrive.retrievers import GoogleDriveRetriever

# Extracted folder ID from the Google Drive URL
folder_id = "1bM-uILBTV_QWmIQ_ohghyrOHfXiZO9zT"

from langchain_community.document_loaders import GoogleDriveLoader

from pathlib import Path

# Initialize the GoogleDriveLoader with the specified paths
loader = GoogleDriveLoader(
    folder_id="1bM-uILBTV_QWmIQ_ohghyrOHfXiZO9zT",  # Replace with your actual folder ID
    credentials_path=Path('/workspaces/Sep10/credentials.json'),
    token_path=Path('/workspaces/Sep10/token.json'),
    recursive=False  # Optional: configure whether to recursively fetch files from subfolders. Defaults to False.
)

# Load documents from the specified Google Drive folder
docs = loader.load()
# Print the loaded documents
print(docs)

