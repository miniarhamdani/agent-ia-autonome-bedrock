"""
Output parsers et structured output (MHD-16).

Objectif : forcer l'agent à produire une décision structurée (JSON valide,
conforme à un schéma Pydantic) plutôt qu'un texte libre — ce qui répond au
Bug 2 du cahier des charges ("Hallucination lors de l'utilisation des outils"),
et complète la limite identifiée en MHD-14 (l'agent simulait des appels
d'outils en prose faute de mécanisme structuré).

Deux niveaux de robustesse :
1. PydanticOutputParser : force le format de sortie via un schéma strict.
2. OutputFixingParser : si le LLM produit un JSON malformé, cette couche
   renvoie automatiquement l'erreur au LLM pour qu'il corrige sa sortie
   (action corrective explicitement demandée par le cahier des charges
   pour le Bug 2), au lieu de planter ou d'halluciner.
"""

import os
import json
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_aws import ChatBedrock
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.prompts import PromptTemplate

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")
MODEL_CLAUDE = os.getenv("BEDROCK_MODEL_CLAUDE", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")


class ToolDecision(BaseModel):
    """Schéma structuré décrivant la décision de l'agent face à une requête."""

    needs_tool: bool = Field(
        description="True si un outil externe est nécessaire pour répondre correctement, False sinon."
    )
    tool_name: Optional[str] = Field(
        default=None,
        description="Nom de l'outil à utiliser (ex. 'search_documents', 'weather_api'). Null si needs_tool est False.",
    )
    tool_input: Optional[dict] = Field(
        default=None,
        description="Paramètres exacts à transmettre à l'outil, sous forme d'objet JSON. Null si needs_tool est False.",
    )
    direct_answer: Optional[str] = Field(
        default=None,
        description="Réponse directe si aucun outil n'est nécessaire. Null si needs_tool est True.",
    )


def build_structured_chain():
    """Construit la chain de décision structurée, avec correction automatique
    des sorties JSON malformées."""
    llm = ChatBedrock(model_id=MODEL_CLAUDE, region_name=AWS_REGION)

    base_parser = PydanticOutputParser(pydantic_object=ToolDecision)
    # OutputFixingParser : en cas de JSON invalide, redemande une correction au LLM
    # au lieu de lever une exception directement (gestion d'erreur du Bug 2).
    fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)

    prompt = PromptTemplate(
        template=(
            "Tu dois analyser la requête utilisateur et décider si un outil externe "
            "est nécessaire pour y répondre.\n\n"
            "Outils disponibles :\n"
            "- search_documents(query: str) : recherche dans la base documentaire interne\n"
            "- weather_api(city: str) : donne la météo actuelle d'une ville\n\n"
            "Réponds STRICTEMENT au format JSON suivant, sans aucun texte avant ou après :\n"
            "{format_instructions}\n\n"
            "Requête utilisateur : {requete}"
        ),
        input_variables=["requete"],
        partial_variables={"format_instructions": base_parser.get_format_instructions()},
    )

    return prompt, llm, fixing_parser


def decide(requete: str) -> ToolDecision:
    prompt, llm, fixing_parser = build_structured_chain()
    formatted_prompt = prompt.format(requete=requete)
    raw_output = llm.invoke(formatted_prompt).content
    decision = fixing_parser.parse(raw_output)
    return decision


if __name__ == "__main__":
    test_cases = [
        "Quelle est notre politique de sécurité IAM pour les projets clients ?",
        "Quel temps fait-il à Tunis aujourd'hui ?",
        "Bonjour, comment vas-tu ?",
    ]

    print("=" * 60)
    print("TEST — Output structuré (ToolDecision)")
    print("=" * 60)

    for requete in test_cases:
        print(f"\nRequête : {requete}")
        try:
            decision = decide(requete)
            print(f"  needs_tool     : {decision.needs_tool}")
            print(f"  tool_name      : {decision.tool_name}")
            print(f"  tool_input     : {decision.tool_input}")
            print(f"  direct_answer  : {decision.direct_answer}")
        except Exception as e:
            print(f"  ÉCHEC — {type(e).__name__}: {e}")
