"""
Chains de raisonnement complexes — Agent ReAct (MHD-17).

Implémente le pattern ReAct (Reasoning and Acting) demandé par le cahier des
charges (US 3.2) : l'agent combine plusieurs outils de façon autonome et
séquentielle pour répondre à une requête complexe, avec des logs explicites
Thought → Action → Observation.

Deux outils sont fournis :
1. search_documents : recherche documentaire interne (stub léger sur un petit
   corpus en mémoire — à remplacer par une vraie base FAISS/ChromaDB si un
   ticket dédié "outil RAG" est créé par la suite).
2. weather_api : appel réel à l'API publique Open-Meteo (gratuite, sans clé),
   pour valider un vrai appel API externe de bout en bout.

Garde-fou anti-boucle infinie (risque MHD-36) : max_iterations est fixé pour
que l'agent s'arrête proprement s'il ne converge pas vers une réponse.
"""

import os
import requests
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain.agents import initialize_agent, AgentType, Tool

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")
MODEL_CLAUDE = os.getenv("BEDROCK_MODEL_CLAUDE", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")

# --- Corpus documentaire interne (stub — à remplacer par FAISS/ChromaDB) ---
CORPUS_INTERNE = {
    "déploiement": (
        "Procédure de déploiement cloud Smartovate : 1) Validation du code sur "
        "une branche feature avec Pull Request revue par un pair. 2) Déploiement "
        "automatique sur l'environnement de staging via CI/CD (GitHub Actions). "
        "3) Tests de non-régression automatisés. 4) Validation manuelle par le "
        "Lead Tech. 5) Déploiement en production via un pipeline blue-green, "
        "avec rollback automatique en cas d'échec des health checks."
    ),
    "sécurité": (
        "Politique de sécurité IAM Smartovate : principe du moindre privilège "
        "systématique, rotation des clés d'accès tous les 90 jours, MFA "
        "obligatoire pour tous les comptes avec accès console, journalisation "
        "CloudTrail activée sur tous les comptes clients."
    ),
}


def search_documents(query: str) -> str:
    """Recherche dans le corpus documentaire interne (stub RAG)."""
    query_lower = query.lower()
    for mot_cle, contenu in CORPUS_INTERNE.items():
        if mot_cle in query_lower:
            return f"[Source interne : {mot_cle}] {contenu}"
    return "Aucun document interne pertinent trouvé pour cette requête."


def weather_api(city: str) -> str:
    """Appel réel à l'API publique Open-Meteo (géocodage + météo actuelle)."""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "fr"},
            timeout=10,
        ).json()
        if not geo.get("results"):
            return f"Ville '{city}' introuvable."
        lat, lon = geo["results"][0]["latitude"], geo["results"][0]["longitude"]

        meteo = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": "true"},
            timeout=10,
        ).json()
        courant = meteo["current_weather"]
        return (
            f"Météo actuelle à {city} : {courant['temperature']}°C, "
            f"vent {courant['windspeed']} km/h."
        )
    except Exception as e:
        return f"Erreur lors de l'appel à l'API météo : {e}"


def build_agent():
    llm = ChatBedrock(model_id=MODEL_CLAUDE, region_name=AWS_REGION)

    tools = [
        Tool(
            name="search_documents",
            func=search_documents,
            description=(
                "Utile pour rechercher des informations dans la documentation interne "
                "de Smartovate (procédures, politiques). Entrée : une requête en texte libre."
            ),
        ),
        Tool(
            name="weather_api",
            func=weather_api,
            description=(
                "Utile pour connaître la météo actuelle d'une ville. "
                "Entrée : le nom de la ville uniquement."
            ),
        ),
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,  # affiche les logs Thought / Action / Observation
        max_iterations=6,  # garde-fou anti-boucle infinie (risque MHD-36)
        handle_parsing_errors=True,  # gestion d'erreur (Bug 2 du cahier des charges)
    )
    return agent


if __name__ == "__main__":
    agent = build_agent()

    # Cas d'usage 5 (MHD-11) : requête combinant RAG + API externe
    requete = (
        "Résume notre procédure de déploiement cloud interne, et dis-moi "
        "s'il fait beau pour la démo client de demain à Tunis."
    )

    print("=" * 60)
    print("TEST — Agent ReAct (raisonnement complexe multi-outils)")
    print("=" * 60)
    print(f"Requête : {requete}\n")

    result = agent.invoke({"input": requete})

    print("\n" + "=" * 60)
    print("RÉPONSE FINALE")
    print("=" * 60)
    print(result["output"])
