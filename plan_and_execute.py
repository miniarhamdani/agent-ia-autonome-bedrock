"""
Agent Plan-and-Execute (MHD-25).

Pattern différent de ReAct (MHD-17/MHD-24) : au lieu de décider une action à
la fois en réagissant à chaque observation (Thought → Action → Observation,
en boucle), l'agent Plan-and-Execute fonctionne en deux temps distincts :

1. PLANIFICATION : un LLM "planificateur" décompose la requête utilisateur en
   une liste d'étapes précises, ÉTABLIE À L'AVANCE, avant toute exécution.
2. EXÉCUTION : chaque étape du plan est exécutée séquentiellement par un
   agent exécutant (qui peut lui-même utiliser des outils), dans l'ordre
   prévu par le plan.

Avantage par rapport à ReAct : plus prévisible et plus efficace sur des
tâches dont la décomposition est évidente dès le départ (moins d'aller-retour
avec le LLM à chaque étape).
Inconvénient : moins adaptatif si un résultat intermédiaire change
fondamentalement ce qu'il faudrait faire ensuite (le plan est fixé au départ).

Réutilise les mêmes outils que MHD-24 (search_documents_vectoriel, weather_api,
query_database) pour permettre une comparaison directe des deux patterns sur
un même socle d'outils.
"""

import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain.agents import Tool
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)

from vector_store import search_documents_vectoriel
from agent_react import weather_api
from custom_tools import query_database

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")
MODEL_CLAUDE = os.getenv("BEDROCK_MODEL_CLAUDE", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")


def build_plan_and_execute_agent() -> PlanAndExecute:
    llm = ChatBedrock(model_id=MODEL_CLAUDE, region_name=AWS_REGION)

    tools = [
        Tool(
            name="search_documents_vectoriel",
            func=search_documents_vectoriel,
            description=(
                "Recherche sémantique dans la documentation interne de Smartovate "
                "(déploiement, sécurité, budget, onboarding, architecture)."
            ),
        ),
        Tool(
            name="weather_api",
            func=weather_api,
            description="Donne la météo actuelle d'une ville. Entrée : le nom de la ville.",
        ),
        Tool(
            name="query_database",
            func=query_database,
            description=(
                "Interroge la base de données des projets Smartovate "
                "(colonnes id, nom, client, statut, budget_usd). Entrée : requête SQL SELECT."
            ),
        ),
    ]

    planificateur = load_chat_planner(llm)
    executeur = load_agent_executor(llm, tools, verbose=True)

    return PlanAndExecute(planner=planificateur, executor=executeur, verbose=True)


if __name__ == "__main__":
    print("=" * 60)
    print("TEST — Agent Plan-and-Execute")
    print("=" * 60)

    agent = build_plan_and_execute_agent()

    # Requête à 3 sous-tâches distinctes et indépendantes — cas idéal pour
    # illustrer l'intérêt d'une planification à l'avance.
    requete = (
        "Fais trois choses : donne-moi le budget total des projets en cours, "
        "rappelle notre politique de sécurité des accès, et dis-moi la météo à Tunis."
    )
    print(f"Requête : {requete}\n")

    result = agent.invoke({"input": requete})

    print("\n" + "=" * 60)
    print("RÉPONSE FINALE")
    print("=" * 60)
    print(result["output"])
