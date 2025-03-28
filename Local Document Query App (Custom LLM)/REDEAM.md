**Step-by-Step Build**
1. Set Up Your Environment
Install Python: Ensure Python 3.8+ is installed.
Create a Virtual Environment:
bash

Collapse

Wrap

Copy
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:
bash

Collapse

Wrap

Copy
pip install streamlit PyPDF2 sentence-transformers qdrant-client openai
Run Qdrant Locally:
Install Docker if not already installed.
Start Qdrant:
bash

Collapse

Wrap

Copy
docker run -p 6333:6333 qdrant/qdrant
Qdrant will be available at http://localhost:6333.
