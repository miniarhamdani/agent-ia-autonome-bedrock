"""
Script de test de connexion à AWS Bedrock (MHD-11 / MHD-14).

Objectif : valider que l'environnement Python + boto3 + LangChain
est correctement configuré et peut invoquer un modèle de fondation
hébergé sur AWS Bedrock (Claude ou Titan).

Usage :
    python test_bedrock_connection.py
"""

import os
import sys
import time
import boto3
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")
MODEL_CLAUDE = os.getenv("BEDROCK_MODEL_CLAUDE", "anthropic.claude-3-5-sonnet-20241022-v2:0")
MODEL_TITAN = os.getenv("BEDROCK_MODEL_TITAN", "amazon.titan-text-express-v1")


def test_boto3_client() -> bool:
    """Vérifie que le client bas niveau boto3 peut lister les modèles Bedrock."""
    print("→ Test 1/3 : connexion boto3 à Bedrock...")
    try:
        client = boto3.client("bedrock", region_name=AWS_REGION)
        response = client.list_foundation_models()
        nb_models = len(response.get("modelSummaries", []))
        print(f"   OK — {nb_models} modèles visibles dans la région {AWS_REGION}.")
        return True
    except Exception as e:
        print(f"   ÉCHEC — {type(e).__name__}: {e}")
        print("   Vérifiez : credentials AWS, région, et permissions IAM (bedrock:ListFoundationModels).")
        return False


def _invoke_model(model_id: str, label: str) -> bool:
    print(f"→ Test : appel du modèle {label} ({model_id})...")
    try:
        from langchain_aws import ChatBedrock

        llm = ChatBedrock(model_id=model_id, region_name=AWS_REGION)
        start = time.time()
        response = llm.invoke("Réponds en une phrase : que fais-tu ?")
        elapsed = time.time() - start

        print(f"   OK — réponse reçue en {elapsed:.2f}s")
        print(f"   Réponse : {response.content[:200]}")
        return True
    except Exception as e:
        print(f"   ÉCHEC — {type(e).__name__}: {e}")
        print("   Vérifiez : le modèle est bien activé dans la console Bedrock, et le rôle IAM a bedrock:InvokeModel.")
        return False


def main():
    print("=" * 60)
    print("TEST DE CONNEXION AWS BEDROCK — Smartovate / MHD-11")
    print("=" * 60)

    results = {
        "boto3_client": test_boto3_client(),
        "claude": _invoke_model(MODEL_CLAUDE, "Claude (Anthropic)"),
        "titan": _invoke_model(MODEL_TITAN, "Titan (Amazon)"),
    }

    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    for name, ok in results.items():
        status = "✅ OK" if ok else "❌ ÉCHEC"
        print(f"  {name:15s} : {status}")

    if not all(results.values()):
        sys.exit(1)

    print("\nTous les tests sont passés. Environnement prêt pour MHD-13/MHD-14.")


if __name__ == "__main__":
    main()
