"""
Contexte long terme via base vectorielle (MHD-23).

Remplace le stub par mots-clés utilisé depuis MHD-17 (`CORPUS_INTERNE`,
recherche par simple correspondance de sous-chaîne) par une vraie recherche
sémantique : les documents sont convertis en vecteurs (embeddings) via le
modèle Bedrock `amazon.titan-embed-text-v2:0`, indexés avec FAISS, puis
recherchés par similarité de sens plutôt que par mot-clé exact.

Différence concrète avec le stub précédent : une requête comme "comment
protéger les accès de nos clients ?" ne contient aucun mot-clé exact du
document de sécurité (qui parle d'"IAM", de "MFA", de "rotation des clés"),
mais la recherche vectorielle le retrouve quand même grâce à la proximité
sémantique — impossible avec une simple correspondance de texte.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_community.vectorstores import FAISS
from langchain.agents import initialize_agent, AgentType, Tool

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")
MODEL_CLAUDE = os.getenv("BEDROCK_MODEL_CLAUDE", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")
MODEL_EMBEDDINGS = os.getenv("BEDROCK_MODEL_EMBEDDINGS", "amazon.titan-embed-text-v2:0")

INDEX_DIR = Path(__file__).parent / "faiss_index_smartovate"

# Corpus documentaire interne étendu (par rapport au stub de MHD-17)
DOCUMENTS = [
    (
        "deploiement",
        "Procédure de déploiement cloud Smartovate : validation du code via Pull "
        "Request revue par un pair, déploiement automatique en staging via CI/CD "
        "GitHub Actions, tests de non-régression automatisés, validation manuelle "
        "par le Lead Tech, puis déploiement en production via un pipeline "
        "blue-green avec rollback automatique en cas d'échec des health checks.",
    ),
    (
        "securite",
        "Politique de sécurité des accès Smartovate : principe du moindre "
        "privilège systématique, authentification à deux facteurs obligatoire "
        "pour tous les comptes avec accès console, rotation des clés d'accès "
        "tous les 90 jours, journalisation complète de toutes les actions "
        "administratives sur les comptes clients.",
    ),
    (
        "budget",
        "Politique budgétaire des projets Smartovate : tout projet dépassant "
        "50 000 USD nécessite une validation du directeur financier avant "
        "démarrage. Les projets internes (R&D, outillage) sont plafonnés à "
        "20 000 USD sauf exception validée en comité de direction.",
    ),
    (
        "onboarding",
        "Processus d'intégration des nouveaux stagiaires et employés : accès "
        "IAM créé sous 48h après signature du contrat, formation obligatoire "
        "sur la charte de sécurité informatique dès le premier jour, binôme "
        "désigné pour les deux premières semaines.",
    ),
    (
        "architecture_agent",
        "Architecture technique de l'agent IA autonome : orchestration via "
        "LangChain, modèles de fondation hébergés sur AWS Bedrock (Claude "
        "Anthropic et Amazon Nova), mémoire conversationnelle persistée sur "
        "DynamoDB, outils connectés via le pattern ReAct avec garde-fou "
        "anti-boucle infinie.",
    ),
]


def construire_index():
    """Construit (ou reconstruit) l'index FAISS à partir du corpus documentaire,
    en générant les embeddings via Bedrock Titan Embeddings."""
    embeddings = BedrockEmbeddings(model_id=MODEL_EMBEDDINGS, region_name=AWS_REGION)
    textes = [contenu for _, contenu in DOCUMENTS]
    metadonnees = [{"source": nom} for nom, _ in DOCUMENTS]

    print(f"Génération des embeddings pour {len(textes)} documents...")
    index = FAISS.from_texts(textes, embeddings, metadatas=metadonnees)
    index.save_local(str(INDEX_DIR))
    print(f"Index FAISS sauvegardé dans {INDEX_DIR}")
    return index


def charger_ou_construire_index():
    embeddings = BedrockEmbeddings(model_id=MODEL_EMBEDDINGS, region_name=AWS_REGION)
    if INDEX_DIR.exists():
        return FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
    return construire_index()


def search_documents_vectoriel(query: str, k: int = 2) -> str:
    """Recherche sémantique dans la base vectorielle — remplace le stub par
    mots-clés utilisé depuis MHD-17."""
    index = charger_ou_construire_index()
    resultats = index.similarity_search_with_score(query, k=k)

    if not resultats:
        return "Aucun document pertinent trouvé."

    sortie = []
    for doc, score in resultats:
        source = doc.metadata.get("source", "inconnue")
        sortie.append(f"[Source : {source}, score de similarité : {score:.3f}] {doc.page_content}")
    return "\n\n".join(sortie)


def build_agent_with_vector_search():
    llm = ChatBedrock(model_id=MODEL_CLAUDE, region_name=AWS_REGION)
    tools = [
        Tool(
            name="search_documents_vectoriel",
            func=search_documents_vectoriel,
            description=(
                "Recherche sémantique dans la documentation interne de Smartovate "
                "(déploiement, sécurité, budget, onboarding, architecture de l'agent). "
                "Fonctionne même sans mot-clé exact, par proximité de sens. "
                "Entrée : une question ou requête en texte libre."
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
    print("TEST 1 — Construction de l'index vectoriel")
    print("=" * 60)
    construire_index()

    print("\n" + "=" * 60)
    print("TEST 2 — Recherche sémantique SANS mot-clé exact")
    print("=" * 60)
    requete = "Comment protège-t-on les accès de nos clients ?"
    print(f"Requête : {requete}")
    print(f"Résultat :\n{search_documents_vectoriel(requete)}")

    print("\n" + "=" * 60)
    print("TEST 3 — Recherche sémantique sur le budget")
    print("=" * 60)
    requete2 = "Quelle somme nécessite l'accord de la direction financière ?"
    print(f"Requête : {requete2}")
    print(f"Résultat :\n{search_documents_vectoriel(requete2)}")

    print("\n" + "=" * 60)
    print("TEST 4 — Agent complet avec recherche vectorielle")
    print("=" * 60)
    agent = build_agent_with_vector_search()
    result = agent.invoke({
        "input": "Combien de temps faut-il pour créer les accès d'un nouveau stagiaire ?"
    })
    print(f"\nRéponse finale : {result['output']}")
