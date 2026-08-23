"""
Outils custom : interrogation de base de données + factory d'outils API (MHD-19).

Répond au périmètre explicite du cahier des charges : "Création d'outils (Tools)
personnalisés pour l'agent (ex: recherche documentaire, interrogation d'une base
de données)".

Deux briques :
1. query_database : interroge une base SQLite de démonstration (table `projets`,
   simulant des données internes Smartovate). Restreint aux requêtes SELECT
   uniquement (sécurité — voir garde-fous ci-dessous, en cohérence avec la
   vigilance déjà appliquée sur MHD-19/MHD-36 pour l'exécution de code).
2. build_api_tool : une factory générique pour créer rapidement un nouvel outil
   d'appel API REST, sans dupliquer le code à chaque fois (contrairement à
   weather_api qui était codé en dur en MHD-17).
"""

import os
import re
import sqlite3
import requests
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain.agents import initialize_agent, AgentType, Tool

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")
MODEL_CLAUDE = os.getenv("BEDROCK_MODEL_CLAUDE", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")

DB_PATH = os.path.join(os.path.dirname(__file__), "smartovate_demo.db")

# --- Garde-fous pour l'outil DB (lecture seule stricte) ---
MOTS_CLES_INTERDITS = {
    "insert", "update", "delete", "drop", "alter", "create", "attach",
    "pragma", "replace", "truncate",
}


class RequeteNonAutoriseeError(Exception):
    """Levée quand la requête SQL n'est pas un SELECT en lecture seule strict."""


def _valider_requete_sql(sql: str) -> None:
    sql_normalise = sql.strip().lower()
    if not sql_normalise.startswith("select"):
        raise RequeteNonAutoriseeError("Seules les requêtes SELECT sont autorisées.")
    if ";" in sql.strip().rstrip(";"):
        raise RequeteNonAutoriseeError("Les requêtes multiples (';') ne sont pas autorisées.")
    mots = set(re.findall(r"[a-zA-Z]+", sql_normalise))
    interdits_trouves = mots & MOTS_CLES_INTERDITS
    if interdits_trouves:
        raise RequeteNonAutoriseeError(f"Mots-clés interdits détectés : {interdits_trouves}")


def query_database(sql: str) -> str:
    """Exécute une requête SELECT en lecture seule sur la base de démonstration."""
    try:
        _valider_requete_sql(sql)
    except RequeteNonAutoriseeError as e:
        return f"REFUSÉ : {e}"

    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)  # ouverture en lecture seule au niveau OS
        cur = conn.cursor()
        cur.execute(sql)
        colonnes = [d[0] for d in cur.description] if cur.description else []
        lignes = cur.fetchmany(50)  # anti déni de service : 50 lignes max
        conn.close()

        if not lignes:
            return "Aucun résultat."
        resultat = " | ".join(colonnes) + "\n"
        resultat += "\n".join(" | ".join(str(v) for v in ligne) for ligne in lignes)
        return resultat
    except Exception as e:
        return f"ÉCHEC lors de l'exécution SQL : {type(e).__name__}: {e}"


def build_api_tool(name: str, description: str, base_url: str, param_name: str, timeout: int = 10) -> Tool:
    """
    Factory générique pour créer un outil d'appel API REST simple (GET avec un
    seul paramètre). Évite de dupliquer le code d'intégration à chaque nouvel
    outil API (comme c'était le cas pour weather_api en MHD-17).

    Exemple :
        currency_tool = build_api_tool(
            name="currency_api",
            description="Donne le taux de change USD vers une devise donnée.",
            base_url="https://api.exchangerate-api.com/v4/latest/USD",
            param_name=None,  # pas de paramètre dans l'URL pour cet exemple
        )
    """

    def _appeler_api(valeur_param: str) -> str:
        try:
            params = {param_name: valeur_param} if param_name else {}
            reponse = requests.get(base_url, params=params, timeout=timeout)
            reponse.raise_for_status()
            return str(reponse.json())[:1500]
        except Exception as e:
            return f"ÉCHEC lors de l'appel à {name} : {type(e).__name__}: {e}"

    return Tool(name=name, func=_appeler_api, description=description)


def build_agent_with_custom_tools():
    llm = ChatBedrock(model_id=MODEL_CLAUDE, region_name=AWS_REGION)

    tools = [
        Tool(
            name="query_database",
            func=query_database,
            description=(
                "Utile pour interroger la base de données interne des projets Smartovate "
                "(table `projets` : colonnes id, nom, client, statut, budget_usd). "
                "Entrée : une requête SQL SELECT valide uniquement (aucune modification autorisée)."
            ),
        ),
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
    print("=" * 60)
    print("TEST 1 — Requête SELECT légitime")
    print("=" * 60)
    print(query_database("SELECT nom, client, statut FROM projets WHERE statut = 'en_cours'"))

    print("\n" + "=" * 60)
    print("TEST 2 — Tentative de DELETE (doit être REFUSÉ)")
    print("=" * 60)
    print(query_database("DELETE FROM projets WHERE id = 1"))

    print("\n" + "=" * 60)
    print("TEST 3 — Tentative d'injection multi-requêtes (doit être REFUSÉ)")
    print("=" * 60)
    print(query_database("SELECT * FROM projets; DROP TABLE projets;"))

    print("\n" + "=" * 60)
    print("TEST 4 — Agent complet utilisant query_database")
    print("=" * 60)
    agent = build_agent_with_custom_tools()
    result = agent.invoke({
        "input": "Quels sont les projets en cours et leur budget total ?"
    })
    print(f"\nRéponse finale : {result['output']}")
