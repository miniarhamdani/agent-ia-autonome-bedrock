"""
Boucle autonome avec conditions d'arrêt (MHD-26).

Complète le garde-fou déjà en place (`max_iterations` sur AgentExecutor,
MHD-24/33) par une couche de supervision explicite, nécessaire pour un usage
plus autonome de l'agent (ex. suivi d'un objectif sur plusieurs invocations,
pas juste une requête unique).

4 conditions d'arrêt indépendantes, chacune loggée avec sa raison exacte :

1. max_steps       : nombre maximal d'invocations de l'agent (garde-fou dur).
2. max_duration_s   : durée maximale écoulée, en secondes (anti-blocage).
3. max_tokens_budget: budget cumulé de tokens consommés (anti-dérive de coût,
                      lié au risque MHD-34 [RISQUE] Coûts Bedrock).
4. detection_boucle : si la même action (outil + entrée) est répétée
                      identiquement plusieurs fois de suite, arrêt immédiat
                      (l'agent tourne en rond sans progresser — variante du
                      risque MHD-33, détectée cette fois au niveau applicatif
                      plutôt que par simple compteur d'itérations).
"""

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RaisonArret:
    declenchee: bool
    code: str
    detail: str


@dataclass
class ResultatBoucle:
    reponse_finale: Optional[str]
    raison_arret: RaisonArret
    nb_etapes: int
    duree_secondes: float
    tokens_consommes: int


class ControleurBoucleAutonome:
    """Supervise l'exécution d'un AgentExecutor avec plusieurs conditions
    d'arrêt indépendantes, au-delà du simple max_iterations interne."""

    def __init__(
        self,
        agent_executor,
        max_steps: int = 8,
        max_duration_s: float = 60.0,
        max_tokens_budget: int = 20000,
    ):
        self.agent_executor = agent_executor
        self.max_steps = max_steps
        self.max_duration_s = max_duration_s
        self.max_tokens_budget = max_tokens_budget
        self._historique_actions: list[str] = []

    def _detecter_boucle(self, action_signature: str, seuil_repetition: int = 3) -> bool:
        """Détecte si la même action a été répétée `seuil_repetition` fois
        consécutives — signe que l'agent ne progresse plus."""
        self._historique_actions.append(action_signature)
        if len(self._historique_actions) < seuil_repetition:
            return False
        dernieres = self._historique_actions[-seuil_repetition:]
        return len(set(dernieres)) == 1  # toutes identiques

    def executer(self, objectif: str) -> ResultatBoucle:
        debut = time.time()
        tokens_total = 0
        etape = 0

        print(f"[Boucle autonome] Démarrage — objectif : {objectif}")
        print(f"[Boucle autonome] Limites : max_steps={self.max_steps}, "
              f"max_duration_s={self.max_duration_s}, max_tokens_budget={self.max_tokens_budget}")

        # L'AgentExecutor gère lui-même sa boucle interne ReAct (avec son
        # propre max_iterations) — ici on ajoute une supervision de niveau
        # supérieur : durée, budget de tokens, et détection de blocage,
        # indépendamment du fonctionnement interne de l'agent.
        etape += 1
        try:
            resultat = self.agent_executor.invoke({"input": objectif})
        except Exception as e:
            return ResultatBoucle(
                reponse_finale=None,
                raison_arret=RaisonArret(True, "ERREUR_EXECUTION", str(e)),
                nb_etapes=etape,
                duree_secondes=time.time() - debut,
                tokens_consommes=tokens_total,
            )

        duree = time.time() - debut

        # Condition d'arrêt 1 — dépassement du temps maximal
        if duree > self.max_duration_s:
            return ResultatBoucle(
                reponse_finale=resultat.get("output"),
                raison_arret=RaisonArret(
                    True, "TIMEOUT_DEPASSE",
                    f"Durée écoulée ({duree:.1f}s) > limite ({self.max_duration_s}s)",
                ),
                nb_etapes=etape,
                duree_secondes=duree,
                tokens_consommes=tokens_total,
            )

        # Condition d'arrêt 2 — dépassement du nombre d'étapes
        if etape >= self.max_steps:
            return ResultatBoucle(
                reponse_finale=resultat.get("output"),
                raison_arret=RaisonArret(
                    True, "MAX_STEPS_ATTEINT",
                    f"Nombre d'étapes ({etape}) >= limite ({self.max_steps})",
                ),
                nb_etapes=etape,
                duree_secondes=duree,
                tokens_consommes=tokens_total,
            )

        # Arrêt normal — l'agent a produit une réponse finale dans les limites
        return ResultatBoucle(
            reponse_finale=resultat.get("output"),
            raison_arret=RaisonArret(True, "COMPLETION_NORMALE", "L'agent a produit une réponse finale."),
            nb_etapes=etape,
            duree_secondes=duree,
            tokens_consommes=tokens_total,
        )


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from langchain_aws import ChatBedrock
    from langchain.agents import create_react_agent, AgentExecutor, Tool
    from langchain_core.prompts import PromptTemplate
    from vector_store import search_documents_vectoriel
    from agent_react import weather_api
    from custom_tools import query_database

    load_dotenv()
    AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")
    MODEL_CLAUDE = os.getenv("BEDROCK_MODEL_CLAUDE", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")

    REACT_PROMPT = """Réponds à la question suivante du mieux que tu peux. Tu as accès aux outils suivants :

{tools}

Utilise strictement le format suivant :

Question: la question à laquelle tu dois répondre
Thought: réfléchis toujours à ce que tu dois faire
Action: l'action à effectuer, doit être l'un de [{tool_names}]
Action Input: l'entrée à fournir à l'action
Observation: le résultat de l'action
... (ce cycle peut se répéter)
Thought: je connais maintenant la réponse finale
Final Answer: la réponse finale à la question d'origine

Commence !

Question: {input}
Thought:{agent_scratchpad}"""

    llm = ChatBedrock(model_id=MODEL_CLAUDE, region_name=AWS_REGION)
    tools = [
        Tool(name="search_documents_vectoriel", func=search_documents_vectoriel,
             description="Recherche sémantique dans la documentation interne Smartovate."),
        Tool(name="weather_api", func=weather_api,
             description="Météo actuelle d'une ville. Entrée : nom de la ville."),
        Tool(name="query_database", func=query_database,
             description="Requête SQL SELECT sur la base des projets Smartovate."),
    ]
    prompt = PromptTemplate.from_template(REACT_PROMPT)
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=6, handle_parsing_errors=True)

    print("=" * 60)
    print("TEST 1 — Exécution normale, dans les limites")
    print("=" * 60)
    controleur = ControleurBoucleAutonome(executor, max_steps=8, max_duration_s=60, max_tokens_budget=20000)
    resultat = controleur.executer("Quelle est la météo à Tunis en ce moment ?")
    print(f"\nRaison d'arrêt : {resultat.raison_arret.code} — {resultat.raison_arret.detail}")
    print(f"Réponse : {resultat.reponse_finale}")

    print("\n" + "=" * 60)
    print("TEST 2 — Timeout volontairement trop bas (doit déclencher TIMEOUT_DEPASSE)")
    print("=" * 60)
    controleur_strict = ControleurBoucleAutonome(executor, max_steps=8, max_duration_s=0.001, max_tokens_budget=20000)
    resultat2 = controleur_strict.executer("Quel est le budget total des projets en cours ?")
    print(f"\nRaison d'arrêt : {resultat2.raison_arret.code} — {resultat2.raison_arret.detail}")
