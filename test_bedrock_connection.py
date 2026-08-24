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
MODEL_NOVA = os.getenv("BEDROCK_MODEL_TITAN", "amazon.nova-lite-v1:0")

# Si des clés d'accès directes sont présentes dans .env, boto3/langchain les
# détecteront automatiquement via les variables d'environnement standard
# AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN.
# Sinon, c'est le profil AWS_PROFILE (SSO) qui sera utilisé.


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
    """Invoque un modèle via l'API Converse de Bedrock (boto3 direct, indépendant
    de la version de langchain-aws — évite les soucis de compatibilité avec les
    modèles récents comme Nova ou les Claude nécessitant un inference profile)."""
    print(f"→ Test : appel du modèle {label} ({model_id})...")
    try:
        client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        start = time.time()
        response = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "Réponds en une phrase : que fais-tu ?"}]}],
        )
        elapsed = time.time() - start

        text = response["output"]["message"]["content"][0]["text"]
        print(f"   OK — réponse reçue en {elapsed:.2f}s")
        print(f"   Réponse : {text[:200]}")
        return True
    except Exception as e:
        print(f"   ÉCHEC — {type(e).__name__}: {e}")
        print("   Vérifiez : le modèle est bien activé, l'ID est correct (inference profile si besoin), et le rôle IAM a bedrock:InvokeModel.")
        return False


def main():
    print("=" * 60)
    print("TEST DE CONNEXION AWS BEDROCK — Smartovate / MHD-11")
    print("=" * 60)

    results = {
        "boto3_client": test_boto3_client(),
        "claude": _invoke_model(MODEL_CLAUDE, "Claude (Anthropic)"),
        "nova": _invoke_model(MODEL_NOVA, "Nova (Amazon)"),
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