#!/bin/bash
# Script simple pour tester le scanner refactorisé

export PYTHONPATH=/home/user/screener/venv/lib/python3.12/site-packages

echo "🧪 TEST DU SCANNER REFACTORISÉ"
echo "================================"
echo ""
echo "Lancement avec config_test.txt (3 symboles: AAPL, NVDA, TSLA)"
echo "Appuyez sur Ctrl+C pour arrêter"
echo ""

python3.12 scanner.py config_test.txt
