#!/bin/bash
# Script pour lancer le scanner en production (tous les symboles)

export PYTHONPATH=/home/user/screener/venv/lib/python3.12/site-packages

echo "📡 SCANNER - MODE PRODUCTION"
echo "============================="
echo ""
echo "Lancement avec config.txt (tous les symboles)"
echo "Appuyez sur Ctrl+C pour arrêter"
echo ""

python3.12 scanner.py config.txt
