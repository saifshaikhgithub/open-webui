from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
load_dotenv()

def get_model():
    model = InferenceClient(model="meta-llama/Meta-Llama-3-70B-Instruct", token=os.environ.get("HF_TOKEN"))
    return model


def run_inference(prompt: str):
    client = get_model()
    try:
        output = client.text_generation(prompt=prompt, max_new_tokens=2048, temperature=0.1)
    except e:
        print(e)
    return output
