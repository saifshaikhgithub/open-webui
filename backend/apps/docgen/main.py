from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from .inference import run_inference
from chromadb.utils import embedding_functions
import chromadb
from .utils import clean_output, get_titles, get_summary

class Prompts(BaseModel):
    base_prompt: str
    step_prompt: str

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_prompt_template = '''list all major sections with no subsections that are needed to be added into a {document_title} with the following description: {document_info}.
Strictly follow the below format for the output:

1. Section Title:

2. Section Title:

3. Section Title:

and so on...
'''

step_prompt = '''The following {document_title} is {document_info}. Only write a section on the {iterating_section} for the {document_title}, Do not write the whole {document_title}. Finally write a short summary of not more than 100 words that encompasses the major details of the section created.
Here is some additional information: {additional_information}

The output format should strictly follow the following format as is, do not remove the "Section Title" or "Section Content" or "Summary", Do not edit the Section Title:

Section Title: {iterating_section}
Section Content: <Content of the section>
Summary: <content of summary>

Make sure to be precise and follow the output format strictly do not say "Here is the output:" in the output.
'''

@app.post("/docgen")
async def docgen(response: Prompts):
    print("Called")
    try:
        document_title = response.base_prompt
        document_info = response.step_prompt

        chroma_client = chromadb.Client()
        embedder = embedding_functions.SentenceTransformerEmbeddingFunction('multi-qa-mpnet-base-cos-v1')

        collection = chroma_client.get_or_create_collection("summaries", embedding_function=embedder)
        init_temp = init_prompt_template
        init_prompt = init_temp.format(document_title=document_title, document_info=document_info)

        output = run_inference(init_prompt)
        print(output)
        titles = get_titles(output)
        final_doc = ""
        counter = 0
        for title in titles:

            temp = step_prompt

            summary = collection.query(
                query_texts=title, n_results=1
            )

            try:
                queried_summary = summary["documents"][0][0]
            except:
                queried_summary = "No additional information found"

            prompt = temp.format(document_title=document_title, document_info=document_info, iterating_section=title, additional_information=queried_summary)

            result = run_inference(prompt)
            parsed_summary = []
            parsed_summary = get_summary(result)

            print("\n\nPrompt Result: ", result)
            print("\n\nSummary from Result Found: ", parsed_summary)
            collection.add(
                f"id{counter}", documents=parsed_summary
            )

            counter += 1
            clean_result = clean_output(result)
            final_doc += clean_result


        print(final_doc)
        return jsonable_encoder({"document": final_doc})
    except Exception as e:
        print("Error: ", e)
        raise HTTPException(status_code=500, detail="e")
