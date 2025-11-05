#!/bin/bash
# Script de test 24/7 - Fonctionne même marché fermé

export PYTHONPATH=/home/user/screener/venv/lib/python3.12/site-packages

echo "🧪 SCANNER EN MODE TEST 24/7"
echo "============================="
echo ""
echo "⚠️  Mode TEST : Ignore les heures de marché"
echo "   Fonctionnera même si le marché est fermé"
echo ""
echo "Config: config_test.txt (AAPL, NVDA, TSLA)"
echo "Appuyez sur Ctrl+C pour arrêter"
echo ""

python3.12 scanner_test_mode.py config_test.txt
