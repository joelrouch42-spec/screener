# 📊 Stock Screener - Documentation

## 🎯 Vue d'ensemble

Système complet de surveillance des marchés financiers avec **deux modes** :
- **🖥️ GUI Dashboard** : Interface web avec graphiques et analyse visuelle
- **⚡ CLI Scanner** : Surveillance continue silencieuse avec alertes uniquement

## 🚀 Démarrage rapide

### Mode GUI Dashboard
```bash
python3 tabs.py [config.txt]
# Interface web accessible sur http://localhost:8050
```

### Mode CLI Scanner
```bash
python3 scanner.py [scanner_config.txt]
# Surveillance continue avec alertes uniquement
```

### Contrôle à distance
```bash
./run_scanner.sh          # Démarrer en arrière-plan
./run_scanner.sh stop     # Arrêter
./run_scanner.sh status   # Vérifier l'état
```

## 📁 Structure des fichiers

### Configuration
- **`settings.json`** : Configuration principale (seuils, market hours, cache, etc.)
- **`config.txt`** : Symboles pour le GUI (format: SYMBOL PROVIDER)
- **`scanner_config.txt`** : Symboles pour le scanner (plus nombreux)
- **`sector_mapping.txt`** : Classification par secteur (SYMBOL SECTOR)

### Scripts principaux
- **`tabs.py`** : Interface GUI avec graphiques Plotly/Dash
- **`scanner.py`** : Scanner CLI optimisé avec cache
- **`catalyst_analyzer.py`** : Analyse des mouvements avec IA optionnelle
- **`data_providers.py`** : Providers multi-sources (Yahoo, Polygon, etc.)

## ⚙️ Configuration avancée

### settings.json - Paramètres clés

```json
{
  "catalyst": {
    "multiplier": 1.5,         // Seuil dynamique = moyenne * multiplier
    "min_threshold": 1.0       // Seuil minimum en %
  },
  "analysis": {
    "volume_spike_threshold": 1.5  // Volume requis (1.5x moyenne)
  },
  "logging": {
    "alerts_only": true        // Mode silencieux pour scanner
  },
  "support_resistance": {
    "order": 5,               // Sensibilité détection S/R
    "cluster_threshold": 0.02  // Regroupement des niveaux (2%)
  }
}
```

### Types d'alertes

#### 🚨 Catalyseurs IA
- **Seuil dynamique** : Basé sur la volatilité historique de chaque action
- **Analyse contextuelle** : News, volume, secteur
- **Classifications** : earnings, technical, contract, macro, etc.

#### 🔧 Breakouts techniques
- **Support breakdown** : Prix casse un support avec volume
- **Resistance breakout** : Prix casse une résistance avec volume
- **Conditions strictes** :
  - Mouvement minimum ±1%
  - Volume spike requis (1.5x moyenne)
  - Close confirmé au-delà du niveau

## 📈 Optimisations performance

### Cache intelligent
- **Niveaux S/R cachés** : Évite le recalcul permanent
- **Données partagées** : GUI et scanner utilisent le même cache
- **Invalidation automatique** : Cache expiré après 1 minute

### Mode silencieux
- **`alerts_only: true`** : Supprime tous les logs techniques
- **Seules les alertes** et erreurs critiques sont affichées
- **Idéal pour surveillance** continue en production

## 🔑 API Claude (Optionnelle)

### Configuration
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
echo 'export ANTHROPIC_API_KEY="sk-ant-..."' >> ~/.bashrc
```

### Fallback automatique
- Sans API key : analyse technique basique
- Avec API key : analyse contextuelle avancée des mouvements

## 🛠️ Exemples d'utilisation

### Surveillance active (marché ouvert)
```bash
# Mode silencieux avec alertes uniquement
python3 scanner.py scanner_config.txt
```

### Analyse graphique détaillée
```bash
# Interface complète avec historique
python3 tabs.py config.txt
```

### Déploiement serveur
```bash
# Démarrage automatique en arrière-plan
./run_scanner.sh
# Logs: tail -f scanner.log
```

## 📊 Interprétation des alertes

### Exemple Breakout
```
🔧 BREAKOUT TECHNIQUE - MSFT
💰 Prix: $515.29 (📉 -2.1%)
🔍 Type: SUPPORT BREAKDOWN
📏 Niveau: $516.70
➡️ Direction: DOWN
⚡ Signal: TECHNIQUE
```
**Signification** : MSFT a cassé son support à $516.70 avec volume, signal baissier.

### Exemple Catalyst
```
🚨 CATALYST DÉTECTÉ - PLTR  
💰 Prix: $194.04 (📉 -6.34%)
🔍 Type: TECHNICAL
⭐ Fiabilité: LOW
💼 Tradeable: ❌ NON
🤖 Signal: INTELLIGENCE ARTIFICIELLE
```
**Signification** : Mouvement significatif détecté mais sans catalyseur clair identifiable.

## 🎛️ Maintenance

### Ajout de nouveaux symboles
1. **GUI** : Modifier `config.txt`
2. **Scanner** : Modifier `scanner_config.txt`  
3. **Secteur** : Ajouter dans `sector_mapping.txt`

### Ajustement des seuils
- **Plus d'alertes** : Réduire `multiplier` dans settings.json
- **Moins de bruit** : Augmenter `min_threshold`
- **Volume** : Ajuster `volume_spike_threshold`

### Debug/Logs
- **Mode verbose** : `"alerts_only": false`
- **Debug providers** : Passer `debug=True` au MultiSourceDataProvider
- **Logs détaillés** : Modifier le niveau logging dans le code

## 🚨 Dépannage

### Erreurs communes
- **Port 8050 occupé** : Changer dans settings.json ou tuer le processus
- **Aucune donnée** : Vérifier providers (Yahoo en fallback)
- **Scanner bloqué** : Vérifier la connectivité réseau
- **Cache corrompu** : Redémarrer l'application

### Performance
- **Scanner lent** : Réduire le nombre de symboles
- **Mémoire élevée** : Augmenter `cache_minutes` dans settings.json
- **CPU élevé** : Augmenter `refresh_seconds`

---

📝 **Version** : Optimisée avec cache intelligent et mode silencieux
🔄 **Dernière MAJ** : 2025-11-04