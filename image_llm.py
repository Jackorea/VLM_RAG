# image_llm.py

from openai import OpenAI
from PIL import Image
from io import BytesIO
import base64
import os

def load_and_encode_image_from_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")
    
    image = Image.open(file_path).convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_base64

def analyze_image(image_path: str, question: str, client, model_name: str) -> str:
    print("🔍 Encodage de l'image...")
    img_base64 = load_and_encode_image_from_file(image_path)

    messages = [
        {"role": "system", "content": "Tu es un assistant capable d'analyser des images."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
            ]
        }
    ]

    print("🚀 Envoi au modèle...")
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.8,
        top_p=0.8
    )

    return response.choices[0].message.content
