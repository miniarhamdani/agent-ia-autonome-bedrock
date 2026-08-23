"""
Mémoire conversationnelle (MHD-21).

Implémente et compare deux stratégies de mémoire LangChain, conformément au
ticket et à l'action corrective prévue pour le Bug 1 du cahier des charges
("Dépassement de la limite de tokens") :

1. ConversationBufferMemory : conserve l'intégralité de l'historique tel quel.
   Simple et fidèle, mais fait croître le nombre de tokens envoyés au modèle
   à chaque tour — c'est justement la cause du Bug 1 sur une conversation longue.

2. ConversationSummaryMemory : résume l'historique au fur et à mesure via le
   LLM lui-même, au lieu de le conserver mot pour mot. Le nombre de tokens
   reste borné même sur une conversation très longue — c'est l'action
   corrective explicitement recommandée par le cahier des charges pour le Bug 1.

Un troisième outil de mesure est fourni pour comparer concrètement le nombre
de tokens utilisés par chaque approche sur une même conversation.
"""

import os
import tiktoken
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")
MODEL_CLAUDE = os.getenv("BEDROCK_MODEL_CLAUDE", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")

# Encodeur pour estimer le nombre de tokens (approximatif pour Claude, mais
# suffisant pour une comparaison relative entre les deux stratégies).
_encodeur = tiktoken.get_encoding("cl100k_base")


def compter_tokens(texte: str) -> int:
    return len(_encodeur.encode(texte))


def get_llm() -> ChatBedrock:
    return ChatBedrock(model_id=MODEL_CLAUDE, region_name=AWS_REGION)


def build_buffer_conversation() -> ConversationChain:
    """Mémoire complète : tout l'historique est conservé tel quel."""
    return ConversationChain(
        llm=get_llm(),
        memory=ConversationBufferMemory(),
        verbose=False,
    )


def build_summary_conversation() -> ConversationChain:
    """Mémoire résumée : l'historique est condensé progressivement par le LLM."""
    return ConversationChain(
        llm=get_llm(),
        memory=ConversationSummaryMemory(llm=get_llm()),
        verbose=False,
    )


# Conversation de test — cas d'usage 2 (MHD-11) : référence à une info donnée
# plusieurs tours plus tôt.
TOURS_DE_TEST = [
    "Je m'appelle Amira et je travaille sur le projet Migration Cloud Alpha, budget 45000 USD.",
    "Le client de ce projet s'appelle Client A.",
    "Peux-tu me rappeler le nom du projet sur lequel je travaille et son budget ?",
    "Et le nom du client, tu t'en souviens ?",
]


def tester_memoire(nom: str, conversation: ConversationChain):
    print(f"\n{'=' * 60}")
    print(f"TEST — {nom}")
    print("=" * 60)

    for i, message in enumerate(TOURS_DE_TEST, start=1):
        reponse = conversation.predict(input=message)
        tokens_memoire = compter_tokens(conversation.memory.buffer if hasattr(conversation.memory, "buffer") else "")
        print(f"\nTour {i} — Utilisateur : {message}")
        print(f"Tour {i} — Agent : {reponse}")
        print(f"[Tokens actuellement stockés en mémoire : ~{tokens_memoire}]")


if __name__ == "__main__":
    tester_memoire("ConversationBufferMemory (historique complet)", build_buffer_conversation())
    tester_memoire("ConversationSummaryMemory (historique résumé)", build_summary_conversation())

    print(f"\n{'=' * 60}")
    print("CONCLUSION")
    print("=" * 60)
    print(
        "Comparez le nombre de tokens affiché après le 4e tour pour chaque "
        "stratégie : ConversationSummaryMemory doit rester significativement "
        "plus compact que ConversationBufferMemory, illustrant la mitigation "
        "du Bug 1 (dépassement de la limite de tokens) sur une conversation longue."
    )
