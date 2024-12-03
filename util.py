import json

def read_json(json_path):
    with open(json_path, 'r') as f:
        embeddings = json.load(f)
    return embeddings
