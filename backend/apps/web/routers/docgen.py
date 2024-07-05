from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from inference import run_inference
from chromadb.utils import embedding_functions
import chromadb
from utils import clean_output, get_titles, get_summary

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def read_file(file):
    with open(file, "r") as f:
        return f.read()


def write_file(file, data):
    with open(file, "w") as f:
        f.write(data)


@app.post("/generatedoc")
def generate_doc(base_prompt: str = Query(..., title="base_prompt"), step_prompt: str = Query(..., title="step_prompt")):
    try:
        chroma_client = chromadb.Client()
        embedder = embedding_functions.SentenceTransformerEmbeddingFunction('multi-qa-mpnet-base-cos-v1')

        collection = chroma_client.get_or_create_collection("summaries", embedding_function=embedder)
        
        output = run_inference(base_prompt)
        titles = get_titles(output)
        final_doc = ""
        counter = 0
        for title in titles:
            temp = step_prompt
            temp = temp.replace("{iterating section}", title)

            summary = collection.query(
                query_texts=title, n_results=1
            )

            queried_summary = summary["documents"][0][0] if summary.get("documents") else "No additional information found"
            
            prompt = temp.replace("{additional information}", queried_summary)
            
            result = run_inference(prompt)
            parsed_summary = get_summary(result)

            print("\n\nPrompt Result: ", result)
            print("\n\nSummary from Result Found: ", parsed_summary)
            collection.add(
                f"id{counter}", documents=parsed_summary
            )

            counter += 1
            clean_result = clean_output(result)
            final_doc += clean_result
        
        return JSONResponse(content=jsonable_encoder({"document": final_doc}))
    
    except Exception as e:
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

