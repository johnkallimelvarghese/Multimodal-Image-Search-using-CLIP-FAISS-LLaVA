import requests
import base64

def ask_llava(image_path, query):
    with open(image_path,"rb") as f:
        image_bytes=f.read()
    image_base64=base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": "llava",
        "prompt": query,
        "images": [image_base64],
        "stream": False,
        "temperature": 0.0,
    }

    response = requests.post("http://localhost:11434/api/generate", json=payload)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return "Error calling LLaVA"
