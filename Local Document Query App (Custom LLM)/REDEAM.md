# Step-by-Step Build

## 1. Set Up Your Environment
   
 ****1.Install Python****: Python 3.8+ 

****2.Create a Virtual Environment:****

On Mac
```
python -m venv venv
source venv/bin/activate  
```
On Windows: 
```
python -m venv venv
venv\Scripts\activate
```

****3.Install Dependencies:****
```
pip install streamlit qdrant-client sentence-transformers PyPDF2 google-generativeai openai requests anthropic
```
****4.Run Qdrant Locally:****
Install Docker if not already installed.
```
docker run -p 6333:6333 qdrant/qdrant
```
Qdrant will be available at http://localhost:6333.


## 2. Run the App
```
streamlit run app.py
```
