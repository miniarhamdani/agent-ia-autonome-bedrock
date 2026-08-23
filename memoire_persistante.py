"""
Mémoire persistante — DynamoDB (MHD-22).

Objectif : faire persister l'historique de conversation AU-DELÀ de l'exécution
d'un seul script Python — condition indispensable pour un déploiement réel sur
AWS Lambda (chaque invocation Lambda est indépendante et sans état ; sans
stockage externe, l'agent "oublierait" tout entre deux messages de l'utilisateur).

DynamoDB a été choisi plutôt que Redis pour deux raisons :
1. Cohérence avec l'architecture déjà 100% AWS du projet (Bedrock, futur
   Lambda/API Gateway) — pas de service tiers supplémentaire à gérer.
2. Mode "pay-per-request" : aucun serveur à maintenir, coût quasi nul pour
   un prototype à faible volume (contrairement à Redis qui nécessite une
   instance ElastiCache active en permanence, facturée H24).

⚠️ Si le rôle IAM/SSO actuel n'a pas les permissions DynamoDB, un mécanisme
de repli local (fichier JSON) est fourni ci-dessous pour ne pas bloquer le
développement en attendant les bons accès (voir USE_LOCAL_FALLBACK).
"""

import os
import json
import boto3
from pathlib import Path
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")
AWS_PROFILE = os.getenv("AWS_PROFILE", "smartovate-dev")
MODEL_CLAUDE = os.getenv("BEDROCK_MODEL_CLAUDE", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")
DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE_NAME", "smartovate-agent-sessions")

# Si définie, pointe vers une instance DynamoDB Local (Docker) au lieu du
# vrai service AWS — permet de tester un vrai comportement DynamoDB sans
# attendre les permissions IAM réelles. Ex. : http://localhost:8000
DYNAMODB_LOCAL_ENDPOINT = os.getenv("DYNAMODB_LOCAL_ENDPOINT", "")

# Passer à False dès que l'accès DynamoDB (local ou réel AWS) est configuré.
USE_LOCAL_FALLBACK = os.getenv("USE_LOCAL_MEMORY_FALLBACK", "true").lower() == "true"

# URL de DynamoDB Local (Docker), si utilisé. Laisser vide pour cibler le
# vrai DynamoDB sur AWS une fois les permissions IAM confirmées.
DYNAMODB_LOCAL_ENDPOINT = os.getenv("DYNAMODB_LOCAL_ENDPOINT", "")

LOCAL_STORE_DIR = Path(__file__).parent / "local_sessions"


class LocalJSONChatMessageHistory(BaseChatMessageHistory):
    """Repli local : persiste l'historique dans un fichier JSON par session.
    Sert de solution temporaire tant que l'accès DynamoDB n'est pas confirmé."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        LOCAL_STORE_DIR.mkdir(exist_ok=True)
        self.fichier = LOCAL_STORE_DIR / f"{session_id}.json"

    @property
    def messages(self):
        if not self.fichier.exists():
            return []
        data = json.loads(self.fichier.read_text(encoding="utf-8"))
        resultat = []
        for m in data:
            if m["type"] == "human":
                resultat.append(HumanMessage(content=m["content"]))
            else:
                resultat.append(AIMessage(content=m["content"]))
        return resultat

    def add_message(self, message) -> None:
        historique = []
        if self.fichier.exists():
            historique = json.loads(self.fichier.read_text(encoding="utf-8"))
        type_msg = "human" if isinstance(message, HumanMessage) else "ai"
        historique.append({"type": type_msg, "content": message.content})
        self.fichier.write_text(json.dumps(historique, ensure_ascii=False, indent=2), encoding="utf-8")

    def clear(self) -> None:
        if self.fichier.exists():
            self.fichier.unlink()


def _get_dynamodb_client():
    """Retourne un client DynamoDB, pointant vers DynamoDB Local (Docker) si
    DYNAMODB_LOCAL_ENDPOINT est défini, sinon vers le vrai service AWS.

    ⚠️ Sans l'option `-sharedDb`, DynamoDB Local isole ses données par
    combinaison région+clé d'accès. On force donc une région fixe ("us-east-1")
    pour tout le mode local, indépendamment de AWS_REGION (qui reste eu-north-1
    pour Bedrock) — sinon la table créée ici et celle utilisée par
    DynamoDBChatMessageHistory finissent dans des "bases" locales différentes."""
    if DYNAMODB_LOCAL_ENDPOINT:
        os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
        return boto3.client(
            "dynamodb",
            endpoint_url=DYNAMODB_LOCAL_ENDPOINT,
            region_name="us-east-1",
            aws_access_key_id="fake",
            aws_secret_access_key="fake",
        )
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    return session.client("dynamodb")


def creer_table_dynamodb_si_absente():
    """Crée la table DynamoDB (locale ou réelle) en mode pay-per-request si
    elle n'existe pas déjà."""
    client = _get_dynamodb_client()
    tables_existantes = client.list_tables()["TableNames"]

    if DYNAMODB_TABLE in tables_existantes:
        print(f"Table '{DYNAMODB_TABLE}' déjà existante.")
        return

    print(f"Création de la table '{DYNAMODB_TABLE}' (pay-per-request)...")
    client.create_table(
        TableName=DYNAMODB_TABLE,
        KeySchema=[{"AttributeName": "SessionId", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "SessionId", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",  # pas de coût fixe, facturé à l'usage réel (AWS réel uniquement)
    )
    print("Table créée. Attente de sa disponibilité...")
    client.get_waiter("table_exists").wait(TableName=DYNAMODB_TABLE)
    print("Table prête.")


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retourne l'historique persistant pour une session donnée — DynamoDB
    (local ou réel) si configuré, sinon repli local JSON."""
    if USE_LOCAL_FALLBACK:
        return LocalJSONChatMessageHistory(session_id)

    if DYNAMODB_LOCAL_ENDPOINT:
        os.environ["AWS_DEFAULT_REGION"] = "us-east-1"  # même région que _get_dynamodb_client
        os.environ.setdefault("AWS_ACCESS_KEY_ID", "fake")
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "fake")
        return DynamoDBChatMessageHistory(
            table_name=DYNAMODB_TABLE,
            session_id=session_id,
            endpoint_url=DYNAMODB_LOCAL_ENDPOINT,
        )

    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    return DynamoDBChatMessageHistory(
        table_name=DYNAMODB_TABLE,
        session_id=session_id,
        boto3_session=session,
    )


def build_persistent_conversation():
    llm = ChatBedrock(model_id=MODEL_CLAUDE, region_name=AWS_REGION)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tu es l'assistant IA de Smartovate Ltd. Réponds de façon claire et concise."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    chain = prompt | llm
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )


if __name__ == "__main__":
    print("=" * 60)
    print(f"TEST — Mémoire persistante ({'repli local JSON' if USE_LOCAL_FALLBACK else 'DynamoDB'})")
    print("=" * 60)

    if not USE_LOCAL_FALLBACK:
        try:
            creer_table_dynamodb_si_absente()
        except Exception as e:
            print(f"\n⚠️ Impossible d'accéder/créer la table DynamoDB : {e}")
            print("→ Vérifiez les permissions IAM de votre rôle SSO auprès de votre encadrant.")
            print("→ En attendant, relancez avec USE_LOCAL_MEMORY_FALLBACK=true dans .env.")
            raise SystemExit(1)

    conversation = build_persistent_conversation()
    session_id = "demo-session-amira"
    config = {"configurable": {"session_id": session_id}}

    print("\n--- Premier message (simule une nouvelle session) ---")
    r1 = conversation.invoke(
        {"input": "Je m'appelle Amira, je travaille sur le projet Migration Cloud Alpha."},
        config=config,
    )
    print(f"Agent : {r1.content}")

    print("\n--- Deuxième message (même session_id, simule la reprise après coupure) ---")
    r2 = conversation.invoke(
        {"input": "Tu te souviens de mon nom et de mon projet ?"},
        config=config,
    )
    print(f"Agent : {r2.content}")

    print("\n" + "=" * 60)
    print("Si l'agent se souvient du nom ET du projet au 2e message, la")
    print("persistance fonctionne — y compris si ce script était relancé")
    print("depuis zéro (nouveau processus Python) avec le même session_id.")
    print("=" * 60)