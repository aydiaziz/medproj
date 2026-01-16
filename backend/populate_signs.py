#!/usr/bin/env python3
"""
Script pour peupler la base de données avec des signes de base de la Langue des Signes Française (LSF)
et autres langues des signes européennes.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from signs_db import create_sign, init_db
import json

# Signes de base LSF (Langue des Signes Française)
# Note: Les landmarks sont des exemples simplifiés. En production, ils devraient être
# capturés depuis des vidéos réelles ou des datasets spécialisés.

LSF_SIGNS = [
    {
        "label": "BONJOUR",
        "aliases": ["salut", "hello", "hi"],
        "landmarks": [
            {"x": 0.5, "y": 0.3, "z": 0.0},  # poignet
            {"x": 0.6, "y": 0.2, "z": 0.0},  # pouce CMC
            {"x": 0.7, "y": 0.15, "z": 0.0}, # pouce MCP
            {"x": 0.75, "y": 0.1, "z": 0.0}, # pouce IP
            {"x": 0.8, "y": 0.05, "z": 0.0}, # pouce tip
            {"x": 0.65, "y": 0.25, "z": 0.0}, # index MCP
            {"x": 0.7, "y": 0.2, "z": 0.0},  # index PIP
            {"x": 0.75, "y": 0.15, "z": 0.0}, # index DIP
            {"x": 0.8, "y": 0.1, "z": 0.0},   # index tip
            {"x": 0.6, "y": 0.3, "z": 0.0},   # middle MCP
            {"x": 0.65, "y": 0.25, "z": 0.0}, # middle PIP
            {"x": 0.7, "y": 0.2, "z": 0.0},   # middle DIP
            {"x": 0.75, "y": 0.15, "z": 0.0}, # middle tip
            {"x": 0.55, "y": 0.35, "z": 0.0}, # ring MCP
            {"x": 0.6, "y": 0.3, "z": 0.0},   # ring PIP
            {"x": 0.65, "y": 0.25, "z": 0.0}, # ring DIP
            {"x": 0.7, "y": 0.2, "z": 0.0},   # ring tip
            {"x": 0.5, "y": 0.4, "z": 0.0},   # pinky MCP
            {"x": 0.55, "y": 0.35, "z": 0.0}, # pinky PIP
            {"x": 0.6, "y": 0.3, "z": 0.0},   # pinky DIP
            {"x": 0.65, "y": 0.25, "z": 0.0}, # pinky tip
        ],
        "metadata": {
            "language": "LSF",
            "difficulty": "easy",
            "category": "salutation"
        }
    },
    {
        "label": "MERCİ",
        "aliases": ["merci", "thank you", "thanks"],
        "landmarks": [
            {"x": 0.5, "y": 0.4, "z": 0.0},   # poignet
            {"x": 0.45, "y": 0.35, "z": 0.0}, # pouce CMC
            {"x": 0.4, "y": 0.3, "z": 0.0},   # pouce MCP
            {"x": 0.35, "y": 0.25, "z": 0.0}, # pouce IP
            {"x": 0.3, "y": 0.2, "z": 0.0},   # pouce tip
            {"x": 0.55, "y": 0.35, "z": 0.0}, # index MCP
            {"x": 0.6, "y": 0.3, "z": 0.0},   # index PIP
            {"x": 0.65, "y": 0.25, "z": 0.0}, # index DIP
            {"x": 0.7, "y": 0.2, "z": 0.0},   # index tip
            {"x": 0.5, "y": 0.4, "z": 0.0},   # middle MCP
            {"x": 0.55, "y": 0.35, "z": 0.0}, # middle PIP
            {"x": 0.6, "y": 0.3, "z": 0.0},   # middle DIP
            {"x": 0.65, "y": 0.25, "z": 0.0}, # middle tip
            {"x": 0.45, "y": 0.45, "z": 0.0}, # ring MCP
            {"x": 0.5, "y": 0.4, "z": 0.0},   # ring PIP
            {"x": 0.55, "y": 0.35, "z": 0.0}, # ring DIP
            {"x": 0.6, "y": 0.3, "z": 0.0},   # ring tip
            {"x": 0.4, "y": 0.5, "z": 0.0},   # pinky MCP
            {"x": 0.45, "y": 0.45, "z": 0.0}, # pinky PIP
            {"x": 0.5, "y": 0.4, "z": 0.0},   # pinky DIP
            {"x": 0.55, "y": 0.35, "z": 0.0}, # pinky tip
        ],
        "metadata": {
            "language": "LSF",
            "difficulty": "easy",
            "category": "politeness"
        }
    },
    {
        "label": "OUI",
        "aliases": ["yes", "oui"],
        "landmarks": [
            {"x": 0.5, "y": 0.5, "z": 0.0},   # poignet
            {"x": 0.5, "y": 0.4, "z": 0.0},   # pouce CMC
            {"x": 0.5, "y": 0.3, "z": 0.0},   # pouce MCP
            {"x": 0.5, "y": 0.2, "z": 0.0},   # pouce IP
            {"x": 0.5, "y": 0.1, "z": 0.0},   # pouce tip
            {"x": 0.6, "y": 0.4, "z": 0.0},   # index MCP
            {"x": 0.65, "y": 0.35, "z": 0.0}, # index PIP
            {"x": 0.7, "y": 0.3, "z": 0.0},   # index DIP
            {"x": 0.75, "y": 0.25, "z": 0.0}, # index tip
            {"x": 0.55, "y": 0.45, "z": 0.0}, # middle MCP
            {"x": 0.6, "y": 0.4, "z": 0.0},   # middle PIP
            {"x": 0.65, "y": 0.35, "z": 0.0}, # middle DIP
            {"x": 0.7, "y": 0.3, "z": 0.0},   # middle tip
            {"x": 0.5, "y": 0.5, "z": 0.0},   # ring MCP
            {"x": 0.55, "y": 0.45, "z": 0.0}, # ring PIP
            {"x": 0.6, "y": 0.4, "z": 0.0},   # ring DIP
            {"x": 0.65, "y": 0.35, "z": 0.0}, # ring tip
            {"x": 0.45, "y": 0.55, "z": 0.0}, # pinky MCP
            {"x": 0.5, "y": 0.5, "z": 0.0},   # pinky PIP
            {"x": 0.55, "y": 0.45, "z": 0.0}, # pinky DIP
            {"x": 0.6, "y": 0.4, "z": 0.0},   # pinky tip
        ],
        "metadata": {
            "language": "LSF",
            "difficulty": "easy",
            "category": "affirmation"
        }
    },
    {
        "label": "NON",
        "aliases": ["no", "non"],
        "landmarks": [
            {"x": 0.5, "y": 0.5, "z": 0.0},   # poignet
            {"x": 0.4, "y": 0.4, "z": 0.0},   # pouce CMC
            {"x": 0.35, "y": 0.35, "z": 0.0}, # pouce MCP
            {"x": 0.3, "y": 0.3, "z": 0.0},   # pouce IP
            {"x": 0.25, "y": 0.25, "z": 0.0}, # pouce tip
            {"x": 0.6, "y": 0.4, "z": 0.0},   # index MCP
            {"x": 0.65, "y": 0.35, "z": 0.0}, # index PIP
            {"x": 0.7, "y": 0.3, "z": 0.0},   # index DIP
            {"x": 0.75, "y": 0.25, "z": 0.0}, # index tip
            {"x": 0.55, "y": 0.45, "z": 0.0}, # middle MCP
            {"x": 0.6, "y": 0.4, "z": 0.0},   # middle PIP
            {"x": 0.65, "y": 0.35, "z": 0.0}, # middle DIP
            {"x": 0.7, "y": 0.3, "z": 0.0},   # middle tip
            {"x": 0.5, "y": 0.5, "z": 0.0},   # ring MCP
            {"x": 0.55, "y": 0.45, "z": 0.0}, # ring PIP
            {"x": 0.6, "y": 0.4, "z": 0.0},   # ring DIP
            {"x": 0.65, "y": 0.35, "z": 0.0}, # ring tip
            {"x": 0.45, "y": 0.55, "z": 0.0}, # pinky MCP
            {"x": 0.5, "y": 0.5, "z": 0.0},   # pinky PIP
            {"x": 0.55, "y": 0.45, "z": 0.0}, # pinky DIP
            {"x": 0.6, "y": 0.4, "z": 0.0},   # pinky tip
        ],
        "metadata": {
            "language": "LSF",
            "difficulty": "easy",
            "category": "negation"
        }
    },
    {
        "label": "AU REVOIR",
        "aliases": ["bye", "goodbye", "au revoir"],
        "landmarks": [
            {"x": 0.5, "y": 0.4, "z": 0.0},   # poignet
            {"x": 0.55, "y": 0.35, "z": 0.0}, # pouce CMC
            {"x": 0.6, "y": 0.3, "z": 0.0},   # pouce MCP
            {"x": 0.65, "y": 0.25, "z": 0.0}, # pouce IP
            {"x": 0.7, "y": 0.2, "z": 0.0},   # pouce tip
            {"x": 0.45, "y": 0.35, "z": 0.0}, # index MCP
            {"x": 0.4, "y": 0.3, "z": 0.0},   # index PIP
            {"x": 0.35, "y": 0.25, "z": 0.0}, # index DIP
            {"x": 0.3, "y": 0.2, "z": 0.0},   # index tip
            {"x": 0.5, "y": 0.4, "z": 0.0},   # middle MCP
            {"x": 0.55, "y": 0.35, "z": 0.0}, # middle PIP
            {"x": 0.6, "y": 0.3, "z": 0.0},   # middle DIP
            {"x": 0.65, "y": 0.25, "z": 0.0}, # middle tip
            {"x": 0.45, "y": 0.45, "z": 0.0}, # ring MCP
            {"x": 0.5, "y": 0.4, "z": 0.0},   # ring PIP
            {"x": 0.55, "y": 0.35, "z": 0.0}, # ring DIP
            {"x": 0.6, "y": 0.3, "z": 0.0},   # ring tip
            {"x": 0.4, "y": 0.5, "z": 0.0},   # pinky MCP
            {"x": 0.45, "y": 0.45, "z": 0.0}, # pinky PIP
            {"x": 0.5, "y": 0.4, "z": 0.0},   # pinky DIP
            {"x": 0.55, "y": 0.35, "z": 0.0}, # pinky tip
        ],
        "metadata": {
            "language": "LSF",
            "difficulty": "medium",
            "category": "salutation"
        }
    }
]

def populate_database():
    """Peupler la base de données avec des signes de base."""
    print("Initialisation de la base de données...")
    init_db()

    print(f"Ajout de {len(LSF_SIGNS)} signes à la base de données...")

    for sign_data in LSF_SIGNS:
        try:
            sign_id = create_sign(
                label=sign_data["label"],
                aliases=sign_data["aliases"],
                landmarks=sign_data["landmarks"],
                metadata=sign_data["metadata"]
            )
            print(f"✓ Ajouté: {sign_data['label']} (ID: {sign_id})")
        except Exception as e:
            print(f"✗ Erreur lors de l'ajout de {sign_data['label']}: {e}")

    print("Population terminée!")

if __name__ == "__main__":
    populate_database()