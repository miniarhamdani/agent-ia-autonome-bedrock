"""
Agent ReAct — implémentation conforme à l'US 3.2 du cahier des charges (MHD-24).

⚠️ Différence importante avec MHD-17 : ce ticket exige explicitement l'usage de
`create_react_agent` (API moderne de LangChain), et non `initialize_agent`
(API historique, dépréciée, utilisée en MHD-17 avec un warning à chaque
exécution). C'est la version officielle et pérenne du pattern ReAct pour
ce projet.

Combine tous les outils développés jusqu'ici :
- search_documents_vectoriel (MHD-23, recherche sémantique FAISS)
- weather_api (MHD-17, appel API externe réel)
- query_database (MHD-20, requête SQL sécurisée en lecture seule)

Critères d'acceptation visés (US 3.2) :
✅ Utilisation de create_react_agent
✅ Logs Thought / Action / Observation explicites
✅ Combinaison autonome de plusieurs outils sur une requête complexe
"""

import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain_core.prompts import PromptTemplate

# Réutilisation des outils déjà développés et testés (évite la duplication)
from vector_store import search_documents_vectoriel
from agent_react import weather_api
from custom_tools import query_database

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")
MODEL_CLAUDE = os.getenv("BEDROCK_MODEL_CLAUDE", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")

# Prompt ReAct classique (format standard popularisé par le papier ReAct /
# repris par LangChain), rédigé manuellement pour ne pas dépendre de
# LangChain Hub (accès réseau externe non garanti en environnement de dev).
REACT_PROMPT_TEMPLATE = """Réponds à la question suivante du mieux que tu peux. Tu as accès aux outils suivants :

{tools}

Utilise strictement le format suivant :

Question: la question à laquelle tu dois répondre
Thought: réfléchis toujours à ce que tu dois faire
Action: l'action à effectuer, doit être l'un de [{tool_names}]
Action Input: l'entrée à fournir à l'action
Observation: le résultat de l'action
... (ce cycle Thought/Action/Action Input/Observation peut se répéter plusieurs fois)
Thought: je connais maintenant la réponse finale
Final Answer: la réponse finale à la question d'origine

Commence !

Question: {input}
Thought:{agent_scratchpad}"""


def build_react_agent() -> AgentExecutor:
    llm = ChatBedrock(model_id=MODEL_CLAUDE, region_name=AWS_REGION)

    tools = [
        Tool(
            name="search_documents_vectoriel",
            func=search_documents_vectoriel,
            description=(
                "Recherche sémantique dans la documentation interne de Smartovate "
                "(déploiement, sécurité, budget, onboarding, architecture). "
                "Entrée : une question ou requête en texte libre."
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
                "Interroge la base de données des projets Smartovate (table `projets` : "
                "id, nom, client, statut, budget_usd). Entrée : une requête SQL SELECT uniquement."
            ),
        ),
    ]

    prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)

    # C'est ici le point exigé par l'US 3.2 : create_react_agent, pas initialize_agent.
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=6,  # garde-fou anti-boucle infinie (MHD-33)
        handle_parsing_errors=True,  # gestion d'erreur (Bug 2 du cahier des charges)
    )


if __name__ == "__main__":
    print("=" * 60)
    print("TEST — Agent ReAct via create_react_agent (US 3.2)")
    print("=" * 60)

    agent_executor = build_react_agent()

    # Requête combinant volontairement 2 outils différents pour valider
    # explicitement le critère "utilisation combinée de plusieurs outils"
    # de l'US 3.2.
    requete = (
        "Quel est le budget total des projets en cours dans notre base de "
        "données, et fait-il beau à Tunis en ce moment ?"
    )
    print(f"Requête : {requete}\n")

    result = agent_executor.invoke({"input": requete})

    print("\n" + "=" * 60)
    print("RÉPONSE FINALE")
    print("=" * 60)
    print(result["output"])
