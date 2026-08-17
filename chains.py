"""
Chains LangChain de base (MHD-15).

Deux chains sont implémentées, comme demandé par le ticket :
1. Une LLMChain simple : répond directement à une question.
2. Une SequentialChain à deux étapes : reformule la question de l'utilisateur
   de façon plus précise, puis génère la réponse à partir de cette version
   reformulée. Utile pour améliorer la qualité des réponses sur des questions
   vagues ou mal formulées.

Note technique : LLMChain et SequentialChain appartiennent à l'API "historique"
de LangChain (chains explicites). Elles sont dépréciées au profit de LCEL
(LangChain Expression Language, syntaxe "|") depuis LangChain 0.1+, mais restent
fonctionnelles dans la version 0.3.7 utilisée sur ce projet, et sont explicitement
demandées par le cahier des charges (US du ticket MHD-15).
"""

import os
import warnings
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")
MODEL_CLAUDE = os.getenv("BEDROCK_MODEL_CLAUDE", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")


def get_llm() -> ChatBedrock:
    """Instancie le client LangChain Bedrock (Claude), partagé par les deux chains."""
    return ChatBedrock(model_id=MODEL_CLAUDE, region_name=AWS_REGION)


def build_simple_chain() -> LLMChain:
    """
    Chain 1 — LLMChain simple.
    Répond directement à une question posée par l'utilisateur.
    """
    prompt = PromptTemplate(
        input_variables=["question"],
        template="Réponds de façon claire et concise à la question suivante : {question}",
    )
    return LLMChain(llm=get_llm(), prompt=prompt, output_key="reponse")


def build_sequential_chain() -> SequentialChain:
    """
    Chain 2 — SequentialChain à deux étapes.
    Étape 1 : reformule la question utilisateur pour la rendre plus précise/complète.
    Étape 2 : génère la réponse à partir de la question reformulée.
    """
    llm = get_llm()

    # Étape 1 : reformulation
    reformulation_prompt = PromptTemplate(
        input_variables=["question_brute"],
        template=(
            "Reformule la question suivante pour qu'elle soit plus précise et complète, "
            "sans changer son sens. Réponds uniquement avec la question reformulée, sans "
            "commentaire additionnel.\n\nQuestion brute : {question_brute}"
        ),
    )
    reformulation_chain = LLMChain(
        llm=llm, prompt=reformulation_prompt, output_key="question_reformulee"
    )

    # Étape 2 : génération de la réponse à partir de la question reformulée
    reponse_prompt = PromptTemplate(
        input_variables=["question_reformulee"],
        template="Réponds de façon claire et structurée à la question suivante : {question_reformulee}",
    )
    reponse_chain = LLMChain(llm=llm, prompt=reponse_prompt, output_key="reponse_finale")

    return SequentialChain(
        chains=[reformulation_chain, reponse_chain],
        input_variables=["question_brute"],
        output_variables=["question_reformulee", "reponse_finale"],
        verbose=True,
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    print("=" * 60)
    print("TEST — Chain 1 : LLMChain simple")
    print("=" * 60)
    simple_chain = build_simple_chain()
    result1 = simple_chain.invoke({"question": "Qu'est-ce que le RAG en IA générative ?"})
    print(f"Réponse : {result1['reponse']}\n")

    print("=" * 60)
    print("TEST — Chain 2 : SequentialChain (reformulation + réponse)")
    print("=" * 60)
    sequential_chain = build_sequential_chain()
    result2 = sequential_chain.invoke({"question_brute": "ça sert à quoi bedrock"})
    print(f"\nQuestion reformulée : {result2['question_reformulee']}")
    print(f"Réponse finale : {result2['reponse_finale']}")
