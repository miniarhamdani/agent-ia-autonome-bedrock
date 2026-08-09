"""
Validation de la classe ChatBedrock de LangChain avec Claude (US 1.2 du cahier des charges).

Ce script complète test_bedrock_connection.py : celui-ci utilisait boto3 directement
(API Converse) pour contourner un souci de compatibilité de langchain-aws avec Nova.
Ici, on valide spécifiquement que LangChain fonctionne bien avec Claude, comme
demandé explicitement par le cahier des charges : "La classe BedrockChat de
LangChain est instanciée avec succès."

Usage :
    python test_langchain_bedrock.py
"""

import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrock

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")
MODEL_CLAUDE = os.getenv("BEDROCK_MODEL_CLAUDE", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")


def main():
    print("=" * 60)
    print("VALIDATION ChatBedrock (LangChain) — US 1.2")
    print("=" * 60)
    print(f"Modèle utilisé : {MODEL_CLAUDE}")
    print(f"Région : {AWS_REGION}\n")

    try:
        llm = ChatBedrock(
            model_id=MODEL_CLAUDE,
            region_name=AWS_REGION,
        )
        print("✅ Classe ChatBedrock instanciée avec succès.\n")

        response = llm.invoke("Réponds en une phrase : que fais-tu ?")
        print("✅ Appel via LangChain réussi.")
        print(f"Réponse : {response.content}")

    except Exception as e:
        print(f"❌ ÉCHEC — {type(e).__name__}: {e}")
        print("\nSi l'erreur mentionne 'on-demand throughput', vérifiez que")
        print("BEDROCK_MODEL_CLAUDE dans .env utilise bien l'ID de l'inference")
        print("profile (préfixe 'eu.' ou 'global.'), pas l'ID direct du modèle.")


if __name__ == "__main__":
    main()
