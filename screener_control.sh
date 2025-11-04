#!/bin/bash

# Script de contrôle à distance pour le screener
# Usage: ./screener_control.sh

CONTROL_FILE=".screener"
APP_SCRIPT="tabs.py"
CONFIG_FILE="config.txt"
PID_FILE=".screener.pid"

log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

is_app_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0  # App tourne
        else
            rm -f "$PID_FILE"  # Nettoyer PID invalide
            return 1  # App arrêtée
        fi
    else
        return 1  # Pas de PID file
    fi
}

start_app() {
    if is_app_running; then
        log "🟢 App déjà active"
        return
    fi
    
    log "🚀 Démarrage de l'application..."
    python3 "$APP_SCRIPT" "$CONFIG_FILE" > screener.log 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_FILE"
    log "🟢 App démarrée (PID: $pid)"
}

stop_app() {
    if ! is_app_running; then
        log "🔴 App déjà arrêtée"
        return
    fi
    
    local pid=$(cat "$PID_FILE")
    log "🛑 Arrêt de l'application (PID: $pid)..."
    
    # Arrêt propre
    kill "$pid" 2>/dev/null
    sleep 2
    
    # Vérifier si encore actif
    if ps -p "$pid" > /dev/null 2>&1; then
        log "🔨 Arrêt forcé..."
        kill -9 "$pid" 2>/dev/null
    fi
    
    # Libérer le port 8050
    fuser -k 8050/tcp 2>/dev/null || true
    
    rm -f "$PID_FILE"
    log "🔴 App arrêtée"
}

cleanup() {
    log "🧹 Nettoyage avant arrêt..."
    stop_app
    exit 0
}

# Gérer Ctrl+C
trap cleanup SIGINT SIGTERM

log "🎛️  Démarrage du contrôleur de screener"
log "📂 Fichier de contrôle: $CONTROL_FILE"
log "💡 Pour contrôler depuis l'extérieur:"
log "   touch $CONTROL_FILE  → démarre l'app"
log "   rm $CONTROL_FILE     → arrête l'app"
log "🔄 Surveillance en cours... (Ctrl+C pour arrêter)"

while true; do
    if [ -f "$CONTROL_FILE" ]; then
        # Fichier présent → app doit tourner
        if ! is_app_running; then
            start_app
        fi
    else
        # Fichier absent → app doit être arrêtée
        if is_app_running; then
            stop_app
        fi
    fi
    
    # Afficher l'état toutes les 10 secondes
    if [ $(($(date +%s) % 10)) -eq 0 ]; then
        if [ -f "$CONTROL_FILE" ]; then
            if is_app_running; then
                log "📊 État: 🟢 ACTIF (fichier: ✅, app: ✅)"
            else
                log "📊 État: 🟠 DÉMARRAGE (fichier: ✅, app: ❌)"
            fi
        else
            log "📊 État: 🔴 ARRÊTÉ (fichier: ❌, app: ❌)"
        fi
    fi
    
    sleep 1
done