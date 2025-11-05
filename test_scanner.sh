#!/bin/bash
# Script de test du scanner refactorisé

echo "🧪 TESTS DU SCANNER REFACTORISÉ"
echo "=" | tr '=' '=' | head -c 80; echo

# Test 1: Tests unitaires
echo "
📋 Test 1: Tests Unitaires"
PYTHONPATH=venv/lib/python3.12/site-packages python3.12 << 'EOF'
from scanner import validate_symbol, validate_provider, RateLimiter, MetricsTracker
import time

print("✅ Imports OK")

# Validation
assert validate_symbol("AAPL") == True
assert validate_symbol("123") == False
assert validate_provider("auto") == True
assert validate_provider("invalid") == False
print("✅ Validation OK")

# RateLimiter
limiter = RateLimiter(max_calls=3, period=1.0)
start = time.time()
for i in range(5):
    limiter.wait_if_needed()
elapsed = time.time() - start
assert elapsed >= 1.5 and elapsed <= 2.5, f"Rate limiter timing incorrect: {elapsed}s"
print(f"✅ RateLimiter OK (waited {elapsed:.2f}s)")

# MetricsTracker
metrics = MetricsTracker()
metrics.record_scan(10.0, 3)
metrics.record_alert(True)
metrics.record_alert(False)
stats = metrics.get_stats()
assert stats['total_scans'] == 1
assert stats['total_alerts'] == 2
print("✅ MetricsTracker OK")

print("\n✅ Tous les tests unitaires passés !")
EOF

if [ $? -ne 0 ]; then
    echo "❌ Tests unitaires échoués"
    exit 1
fi

# Test 2: Vérifier la configuration
echo "
📋 Test 2: Configuration"
if [ ! -f "config_test.txt" ]; then
    echo "❌ config_test.txt introuvable"
    exit 1
fi
echo "✅ config_test.txt trouvé"

if [ ! -f "settings.json" ]; then
    echo "❌ settings.json introuvable"
    exit 1
fi
echo "✅ settings.json trouvé"

# Test 3: Lancer le scanner (dry-run 5 secondes)
echo "
📋 Test 3: Lancer le scanner (5 secondes)"
echo "Note: Le marché doit être ouvert pour voir des scans"
echo ""

timeout 5 PYTHONPATH=venv/lib/python3.12/site-packages python3.12 scanner.py config_test.txt 2>&1 | head -30

echo ""
echo "=" | tr '=' '=' | head -c 80; echo
echo "✅ TESTS TERMINÉS"
echo ""
echo "Pour lancer le scanner complet:"
echo "  PYTHONPATH=venv/lib/python3.12/site-packages python3.12 scanner.py config_test.txt"
echo ""
echo "Pour le production (tous les symboles):"
echo "  PYTHONPATH=venv/lib/python3.12/site-packages python3.12 scanner.py config.txt"
