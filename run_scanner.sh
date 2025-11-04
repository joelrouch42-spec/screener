#!/bin/bash

# Script de lancement du scanner
# Usage: ./run_scanner.sh [config_file]

CONFIG_FILE="${1:-scanner_config.txt}"
LOG_FILE="scanner.log"
PID_FILE="scanner.pid"

start_scanner() {
    if [ -f "$PID_FILE" ] && ps -p $(cat "$PID_FILE") > /dev/null 2>&1; then
        echo "🔴 Scanner déjà en cours d'exécution (PID: $(cat $PID_FILE))"
        return 1
    fi
    
    echo "🚀 Démarrage du scanner avec $CONFIG_FILE..."
    
    # Lancer en arrière-plan avec logs
    python3 scanner.py "$CONFIG_FILE" > "$LOG_FILE" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_FILE"
    
    echo "✅ Scanner démarré (PID: $pid)"
    echo "📋 Logs: tail -f $LOG_FILE"
    echo "🛑 Arrêter: ./run_scanner.sh stop"
}

stop_scanner() {
    if [ ! -f "$PID_FILE" ]; then
        echo "🔴 Aucun scanner en cours"
        return 1
    fi
    
    local pid=$(cat "$PID_FILE")
    if ps -p "$pid" > /dev/null 2>&1; then
        echo "🛑 Arrêt du scanner (PID: $pid)..."
        kill "$pid"
        
        # Attendre l'arrêt
        sleep 2
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "🔨 Arrêt forcé..."
            kill -9 "$pid"
        fi
        
        rm -f "$PID_FILE"
        echo "✅ Scanner arrêté"
    else
        echo "🔴 Scanner non trouvé"
        rm -f "$PID_FILE"
    fi
}

status_scanner() {
    if [ -f "$PID_FILE" ] && ps -p $(cat "$PID_FILE") > /dev/null 2>&1; then
        local pid=$(cat "$PID_FILE")
        echo "🟢 Scanner actif (PID: $pid)"
        echo "📊 Dernières lignes du log:"
        tail -5 "$LOG_FILE" 2>/dev/null || echo "   (Pas de logs)"
    else
        echo "🔴 Scanner arrêté"
        [ -f "$PID_FILE" ] && rm -f "$PID_FILE"
    fi
}

case "$1" in
    stop)
        stop_scanner
        ;;
    status)
        status_scanner
        ;;
    restart)
        stop_scanner
        sleep 1
        start_scanner
        ;;
    *)
        start_scanner
        ;;
esac