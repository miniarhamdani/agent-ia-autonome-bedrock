# Agent IA Autonome — AWS Bedrock + LangChain

Prototype d'agent IA développé par Smartovate Ltd, s'appuyant sur AWS Bedrock
(Claude, Titan) pour les modèles de fondation et LangChain pour l'orchestration.

## Prérequis

- Python 3.10 ou supérieur
- Un compte AWS avec accès à Bedrock activé (voir ticket MHD-13)
- AWS CLI configuré (`aws configure`) ou un profil IAM valide

## Installation

```bash
# 1. Cloner le dépôt
git clone <url-du-repo>
cd <nom-du-repo>

# 2. Créer et activer l'environnement virtuel
python3.10 -m venv venv
source venv/bin/activate      # Sous Windows : venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer les variables d'environnement
cp .env.example .env
# Puis éditer .env avec vos valeurs (région AWS, profil, IDs de modèles)
```

## Vérifier l'installation

```bash
python test_bedrock_connection.py
```

Ce script vérifie :
1. La connexion boto3 à Bedrock (liste des modèles disponibles)
2. L'invocation du modèle Claude via LangChain
3. L'invocation du modèle Titan via LangChain

## Structure du projet (à venir au fil des sprints)

```
.
├── requirements.txt
├── .env.example
├── .gitignore
├── test_bedrock_connection.py
├── prompts/              # System prompts versionnés (MHD-15)
├── tools/                # Tools LangChain (RAG, API externe...)
└── app/                  # Interface Streamlit / API FastAPI (Sprint 4)
```

## Versions des dépendances principales

Voir `requirements.txt`. Après installation, générer le lock exact avec :

```bash
pip freeze > requirements-lock.txt
```
