import os
from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()
OpenAI.api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI()

def load_texts(directory):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                texts.append(file.read())
    return texts

def get_embeddings(texts):
    embeddings = []
    for text in texts:
        response = client.embeddings.create(
            input = text,
            model ="text-embedding-3-small"
        )
        embeddings.append(response.data[0].embedding)    
    return embeddings


texts = load_texts('./archive/business')
embeddings = get_embeddings(texts)

# print("Finish get embedding")

with open('embeddings.json', 'w') as f:
    json.dump(embeddings, f)




