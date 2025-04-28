# from openai import OpenAI
# from langchain_chroma import Chroma
# from langchain.prompts import ChatPromptTemplate

# from get_embedding_function import get_embedding_function

# positive_keywords = ["priorité", "signalisation", "panneau", "intersection", "circulation", "voie", "stop", "feu tricolore"]
# negative_keywords = ["immobilisation", "remorquage", "accident", "dépannage", "bris", "enlèvement", "panne", "fourrière", "infractions graves"]

# def filter_documents(docs):
#     filtered_docs = []
#     for doc in docs:
#         content = doc.page_content.lower()
#         if any(pk in content for pk in positive_keywords) and not any(nk in content for nk in negative_keywords):
#             filtered_docs.append(doc)
#     return filtered_docs


# CHROMA_PATH = "chroma"

# PROMPT_TEMPLATE = """
# Vous êtes un expert du Code de la Route et un observateur de scènes visuelles.

# Votre décision (GO ou STOP) doit être basée **prioritairement** sur l'analyse directe de la scène. 
# La recherche documentaire ci-dessous (issu du Code de la Route) doit uniquement servir à vérifier ou renforcer votre décision initiale, **sans la contredire** si la scène est claire et sans danger visible.

# {context}

# ---

# Answer the question based on the above context: {question}
# """


# # ✅ Initialize OpenAI Client globally
# client = OpenAI(
#     api_key="FXMLo8lTdaSqsmDDRuJ3mT55xcJCryQy",
#     base_url="https://llm.intellisphere.fr:9081/v1",
#     timeout=60
# )

# # ✅ Hardcode the model name
# model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

# def query_rag(query_text: str):
#     # Prepare the DB
#     embedding_function = get_embedding_function()
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

#     # Search the DB
#     results = db.similarity_search_with_score(query_text, k=5)

#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     prompt = prompt_template.format(context=context_text, question=query_text)

#     # 🛠️ Use OpenAI client directly, not OllamaLLM
#     response = client.chat.completions.create(
#         model=model_name,
#         messages=[
#             {"role": "system", "content": "Tu es un assistant expert en droit du Code de la Route."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.7,
#         top_p=0.9
#     )

#     response_text = response.choices[0].message.content

#     sources = [doc.metadata.get("id", None) for doc, _score in results]
#     formatted_response = f"Response: {response_text}\nSources: {sources}"
#     print(formatted_response)
#     return response_text



# if __name__ == "__main__":
#     main()

# query_data.py

from openai import OpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

# PROMPT_TEMPLATE = """
# Vous êtes un expert du Code de la Route et un observateur de scènes visuelles.

# Voici l'analyse préliminaire d'un modèle de vision sur une scène :
# "{vision_response}"

# Basé sur cette analyse et en utilisant uniquement les extraits documentaires suivants issus du Code de la Route :

# {context}

# ---

# Décidez si le véhicule peut partir en répondant uniquement par 'GO' ou 'STOP'. 
# Justifiez brièvement en citant une règle applicable, en confirmant ou infirmant l'analyse préliminaire si nécessaire.
# """

PROMPT_TEMPLATE = """
Vous êtes un expert du Code de la Route et un observateur de scènes visuelles.

Voici la description d'une scène routière observée depuis la perspective du conducteur :
"{vision_response}"

En vous basant uniquement sur cette description visuelle et sur les extraits documentaires suivants issus du Code de la Route :

{context}

---

Votre tâche :
- Décidez si le véhicule peut partir (**GO**) ou doit rester arrêté (**STOP**) ou "Démarrer lentement".
- Justifiez votre décision en citant une règle applicable du Code de la Route.

Format de réponse :
- Commencez par **GO** ou **STOP** ou "Démarrer lentement"(en majuscules).
- Ajoutez une justification très courte (1 à 2 phrases maximum).

Important :
- N'inventez pas d'éléments absents de la description.
- Ne donnez jamais une réponse sans justification légale ou observation logique basée sur la scène.
"""



client = OpenAI(
    api_key="FXMLo8lTdaSqsmDDRuJ3mT55xcJCryQy",
    base_url="https://llm.intellisphere.fr:9081/v1",
    timeout=60
)

model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

def query_rag(vision_response: str):
    # 🔎 Préparation de la base de données
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # 📚 Recherche dans Chroma basée sur la réponse vision
    results = db.similarity_search_with_score(vision_response, k=5)

    # 🧩 Construction du contexte
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # 📝 Création du prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, vision_response=vision_response)

    # 🚀 Envoi à LLM pour finaliser la réponse
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Tu es un assistant expert en droit du Code de la Route."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        top_p=0.9
    )

    # 📜 Extraction de la réponse
    response_text = response.choices[0].message.content

    # 📑 Extraction des sources
    sources = [doc.metadata.get("id", None) for doc, _score in results if doc.metadata.get("id", None)]

    # 🎯 Formatage final
    formatted_response = f"{response_text}\n\n🔗 Sources utilisées : {sources if sources else 'Aucune source disponible.'}"
    
    print(formatted_response)
    return formatted_response

if __name__ == "__main__":
    raise Exception("Ce script n'est pas censé être exécuté directement.")
