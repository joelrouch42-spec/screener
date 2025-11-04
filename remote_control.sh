#!/bin/bash

# Script de contrôle local pour le screener distant
# Usage: ./remote_control.sh [start|stop|status]

REMOTE_CONTROL_FILE="/mnt/ssh_mount/.screener"

case "$1" in
    start)
        echo "🚀 Démarrage du screener distant..."
        touch "$REMOTE_CONTROL_FILE"
        echo "✅ Commande envoyée"
        ;;
    stop)
        echo "🛑 Arrêt du screener distant..."
        rm -f "$REMOTE_CONTROL_FILE"
        echo "✅ Commande envoyée"
        ;;
    status)
        if [ -f "$REMOTE_CONTROL_FILE" ]; then
            echo "📊 État: 🟢 ACTIF (fichier de contrôle présent)"
        else
            echo "📊 État: 🔴 ARRÊTÉ (fichier de contrôle absent)"
        fi
        ;;
    *)
        echo "Usage: $0 [start|stop|status]"
        echo ""
        echo "  start   - Démarre le screener distant"
        echo "  stop    - Arrête le screener distant" 
        echo "  status  - Affiche l'état du screener"
        exit 1
        ;;
esac