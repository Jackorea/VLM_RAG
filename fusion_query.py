# # fusion_query.py

# from openai import OpenAI
# from image_llm import analyze_image
# from query_data import query_rag

# def main():
#     print("🔥 fusion_query.py started")

#     image_path = "Carla_image/Town01_001140.png"
#     question = """
#     Basé uniquement sur la scène fournie et en supposant que le véhicule est en parfait état de fonctionnement, 
#     indiquez si le véhicule peut partir en répondant obligatoirement par 'GO' ou 'STOP'. 
#     Justifiez brièvement en citant une règle applicable du 'Code de la Route'. 
#     Basez la décision uniquement sur les éléments visibles. 
#     """
#     # ✅ Initialize client here
#     client = OpenAI(
#         api_key="FXMLo8lTdaSqsmDDRuJ3mT55xcJCryQy",
#         base_url="https://llm.intellisphere.fr:9081/v1",
#         timeout=60
#     )
#     model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

#     try:
#         print("🔍 Analyse de l'image via LLM...")
#         image_response = analyze_image(image_path, question, client, model_name)
#         print("\n🧠 Réponse du modèle sur l’image :")
#         print(image_response)

#         print("\n📚 Recherche contextuelle via RAG...")
#         rag_response = query_rag(question)
#         print("\n📎 Réponse RAG :")
#         print(rag_response)

#     except Exception as e:
#         print(f"💥 Une erreur est survenue : {e}")

# if __name__ == "__main__":
#     main()


# fusion_query.py

from openai import OpenAI
from image_llm import analyze_image
from query_data import query_rag

def main():
    print("🔥 fusion_query.py started")

    image_path = "Carla_image/traffic_sign_30.png"
    question = """
    Voici une image capturée depuis la perspective du conducteur d'un véhicule.

    Décris précisément la scène visible :
    - Quels sont les éléments de signalisation (panneaux, feux, marquages) ?
    - Quels sont les usagers de la route présents (piétons, cyclistes, autres véhicules) ?
    - Quelle est leur position relative par rapport au véhicule ?
    - Y a-t-il des obstacles ou dangers visibles ?
    """
    
    # """
    # Basé uniquement sur la scène fournie et en supposant que le véhicule est en parfait état de fonctionnement, 
    # indiquez si le véhicule peut partir en répondant obligatoirement par 'GO' ou 'STOP'. 
    # Justifiez brièvement en citant une règle applicable du 'Code de la Route'. 
    # Basez la décision uniquement sur les éléments visibles. 
    # """
    
    # ✅ Initialize OpenAI client
    client = OpenAI(
        api_key="FXMLo8lTdaSqsmDDRuJ3mT55xcJCryQy",
        base_url="https://llm.intellisphere.fr:9081/v1",
        timeout=60
    )
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

    try:
        print("🔍 Analyse de l'image via LLM...")
        image_response = analyze_image(image_path, question, client, model_name)
        print("\n🧠 Réponse du modèle sur l’image :")
        print(image_response)

        print("\n📚 Recherche contextuelle via RAG avec la réponse vision...")
        final_response = query_rag(image_response)
        print("\n✅ Réponse finale après enrichissement RAG :")
        print(final_response)

    except Exception as e:
        print(f"💥 Une erreur est survenue : {e}")

if __name__ == "__main__":
    main()
