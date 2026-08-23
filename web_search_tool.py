"""
Outil de recherche web — Tavily (MHD-18).

Ajoute une capacité de recherche web en temps réel à l'agent, en complément
de la recherche documentaire interne (search_documents) et de l'appel API
météo déjà implémentés en MHD-17.

Tavily a été choisi plutôt que SerpAPI : intégration native dans LangChain
(langchain_community.tools.tavily_search), résultats déjà synthétisés
(pas de HTML brut à parser), et tier gratuit adapté à un prototype
(1000 requêtes/mois).

Prérequis : une clé API Tavily gratuite sur https://tavily.com
(inscription en 1 minute, pas de carte bancaire requise pour le tier gratuit).
"""

import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import initialize_agent, AgentType, Tool

# Import des outils déjà développés en MHD-17, pour les combiner ensemble
from agent_react import search_documents, weather_api

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")
MODEL_CLAUDE = os.getenv("BEDROCK_MODEL_CLAUDE", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


def build_web_search_tool() -> TavilySearchResults:
    """Construit l'outil de recherche web Tavily."""
    if not TAVILY_API_KEY:
        raise ValueError(
            "TAVILY_API_KEY manquante dans .env. "
            "Créez une clé gratuite sur https://tavily.com puis ajoutez-la à votre .env."
        )
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY  # requis par la lib tavily-python
    return TavilySearchResults(
        max_results=3,
        description=(
            "Utile pour rechercher des informations récentes ou publiques sur le web "
            "(actualités, documentation publique, informations générales non internes "
            "à Smartovate). Entrée : une requête de recherche en texte libre."
        ),
    )


def build_agent_with_web_search():
    """Construit l'agent en combinant les 3 outils : RAG interne, météo, et
    recherche web — pour montrer la montée en puissance progressive de l'agent."""
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
            description="Utile pour connaître la météo actuelle d'une ville. Entrée : le nom de la ville.",
        ),
        build_web_search_tool(),
    ]

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=6,
        handle_parsing_errors=True,
    )


if __name__ == "__main__":
    agent = build_agent_with_web_search()

    # Requête nécessitant spécifiquement une info publique récente (web),
    # pas couverte par le corpus interne ni par la météo.
    requete = "Quelles sont les dernières nouveautés annoncées sur AWS Bedrock en 2026 ?"

    print("=" * 60)
    print("TEST — Outil de recherche web (Tavily)")
    print("=" * 60)
    print(f"Requête : {requete}\n")

    result = agent.invoke({"input": requete})

    print("\n" + "=" * 60)
    print("RÉPONSE FINALE")
    print("=" * 60)
    print(result["output"])
