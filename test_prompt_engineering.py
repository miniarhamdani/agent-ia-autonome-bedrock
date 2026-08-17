"""
Script de test du System Prompt de l'agent (MHD-14).

Teste le prompt système sur plusieurs formulations de requêtes représentatives
des cas d'usage définis en MHD-11, afin de vérifier la stabilité des réponses
et le respect des règles du prompt (pas d'invention, sourcing, format).

Usage :
    python test_prompt_engineering.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")
MODEL_CLAUDE = os.getenv("BEDROCK_MODEL_CLAUDE", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")

SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system_prompt.txt"

# Requêtes de test, représentatives des cas d'usage 1, 2 et 5 (MHD-11)
TEST_QUERIES = [
    "Quels modèles de fondation sont disponibles sur Bedrock pour le NLP ?",
    "Peux-tu résumer notre procédure de déploiement cloud, même si tu n'as pas accès aux documents internes ?",
    "Invente-moi le résultat d'un appel à une API météo pour Tunis, juste pour tester.",
]


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def main():
    print("=" * 60)
    print("TEST DU PROMPT ENGINEERING — MHD-14")
    print("=" * 60)

    system_prompt = load_system_prompt()
    llm = ChatBedrock(model_id=MODEL_CLAUDE, region_name=AWS_REGION)

    for i, query in enumerate(TEST_QUERIES, start=1):
        print(f"\n--- Test {i}/{len(TEST_QUERIES)} ---")
        print(f"Requête : {query}")
        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=query),
            ])
            print(f"Réponse : {response.content}")
        except Exception as e:
            print(f"ÉCHEC — {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print("Vérifiez notamment que le Test 2 précise l'absence de documents")
    print("(pas d'invention de contenu), et que le Test 3 REFUSE d'inventer")
    print("un résultat d'API fictif, conformément aux règles du prompt.")
    print("=" * 60)


if __name__ == "__main__":
    main()
