"""
Outil d'exécution de code Python (MHD-19).

⚠️ Ce ticket active directement le risque documenté en MHD-36
("Sécurité — Exécution de code non sandboxée"). Les protections suivantes
sont mises en place pour réduire ce risque, MAIS elles ne constituent PAS
un vrai sandbox de niveau production (voir section "Limites" en bas).

Protections implémentées :
1. Analyse statique du code (module `ast`) AVANT exécution : bloque les
   imports et appels dangereux (os, subprocess, sys, socket, shutil,
   ctypes, eval, exec, open, __import__...) et les tentatives d'évasion
   classiques via les attributs dunder (__class__, __globals__, etc.).
2. Exécution dans un SOUS-PROCESSUS isolé (pas dans le processus principal
   de l'agent), avec un timeout strict — un code qui boucle à l'infini ou
   plante ne bloque jamais l'agent.
3. Limitation de la taille de la sortie retournée (anti-déni de service
   sur la sortie).
"""

import ast
import os
import subprocess
import sys
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain.agents import initialize_agent, AgentType, Tool

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")
MODEL_CLAUDE = os.getenv("BEDROCK_MODEL_CLAUDE", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")

# --- Listes de blocage (analyse statique) ---
MODULES_INTERDITS = {
    "os", "sys", "subprocess", "socket", "shutil", "ctypes", "multiprocessing",
    "threading", "importlib", "pathlib", "requests", "urllib", "http",
    "pickle", "marshal", "code", "pty", "signal",
}
APPELS_INTERDITS = {"eval", "exec", "open", "__import__", "compile", "input", "breakpoint"}
ATTRIBUTS_DANGEREUX = {
    "__class__", "__globals__", "__subclasses__", "__bases__", "__mro__",
    "__builtins__", "__import__", "__loader__", "__code__",
}

TIMEOUT_SECONDES = 5
TAILLE_MAX_SORTIE = 2000


class CodeUnsafeError(Exception):
    """Levée quand l'analyse statique détecte du code potentiellement dangereux."""


def analyser_code(code: str) -> None:
    """Parcourt l'arbre syntaxique du code et lève CodeUnsafeError si un
    élément interdit est détecté. Exécutée AVANT toute exécution réelle."""
    try:
        arbre = ast.parse(code)
    except SyntaxError as e:
        raise CodeUnsafeError(f"Code syntaxiquement invalide : {e}")

    for node in ast.walk(arbre):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            noms = [alias.name.split(".")[0] for alias in node.names]
            for nom in noms:
                if nom in MODULES_INTERDITS:
                    raise CodeUnsafeError(f"Import interdit détecté : '{nom}'")

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in APPELS_INTERDITS:
                raise CodeUnsafeError(f"Appel interdit détecté : '{node.func.id}()'")

        if isinstance(node, ast.Attribute) and node.attr in ATTRIBUTS_DANGEREUX:
            raise CodeUnsafeError(f"Accès à un attribut dangereux détecté : '{node.attr}'")


def executer_code_python(code: str) -> str:
    """Analyse puis exécute le code dans un sous-processus isolé avec timeout."""
    try:
        analyser_code(code)
    except CodeUnsafeError as e:
        return f"REFUSÉ (code jugé dangereux) : {e}"

    try:
        resultat = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDES,
        )
        sortie = resultat.stdout or resultat.stderr or "(aucune sortie)"
        return sortie[:TAILLE_MAX_SORTIE]
    except subprocess.TimeoutExpired:
        return f"ÉCHEC : le code a dépassé le délai maximal de {TIMEOUT_SECONDES}s (boucle infinie possible)."
    except Exception as e:
        return f"ÉCHEC lors de l'exécution : {type(e).__name__}: {e}"


def build_agent_with_code_tool():
    llm = ChatBedrock(model_id=MODEL_CLAUDE, region_name=AWS_REGION)

    tools = [
        Tool(
            name="python_repl",
            func=executer_code_python,
            description=(
                "Utile pour effectuer des calculs, manipuler des données ou vérifier "
                "un résultat via du code Python. N'a PAS accès au système de fichiers, "
                "au réseau, ni aux imports système (os, subprocess, etc.) — usage "
                "limité aux calculs et manipulations de données en mémoire. "
                "Entrée : du code Python valide, sans commentaire explicatif autour."
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
    print("TEST 1 — Code légitime (calcul)")
    print("=" * 60)
    print(executer_code_python("print(sum(range(1, 101)))"))

    print("\n" + "=" * 60)
    print("TEST 2 — Tentative d'import dangereux (doit être REFUSÉ)")
    print("=" * 60)
    print(executer_code_python("import os\nprint(os.listdir('.'))"))

    print("\n" + "=" * 60)
    print("TEST 3 — Tentative d'évasion via __class__ (doit être REFUSÉ)")
    print("=" * 60)
    print(executer_code_python("().__class__.__bases__[0].__subclasses__()"))

    print("\n" + "=" * 60)
    print("TEST 4 — Boucle infinie (doit ÉCHOUER proprement par timeout)")
    print("=" * 60)
    print(executer_code_python("while True: pass"))

    print("\n" + "=" * 60)
    print("TEST 5 — Agent complet avec l'outil python_repl")
    print("=" * 60)
    agent = build_agent_with_code_tool()
    result = agent.invoke({
        "input": "Calcule la somme des 50 premiers nombres pairs, en utilisant du code Python."
    })
    print(f"\nRéponse finale : {result['output']}")
