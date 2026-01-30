SAFRAN — Analyse Boursière 2025

Application Streamlit dédiée à l’analyse des données boursières de l’entreprise Safran sur l’année 2025, réalisée dans un cadre pédagogique.

👩‍🎓 Étudiante : Assia BOUDJRAF
🎓 Formation : BUT 3 Science des Données VCOD

🎯 Objectifs du projet

Mettre en pratique les compétences en analyse de données financières

Concevoir une application interactive de visualisation avec Streamlit

Appliquer des indicateurs techniques boursiers

Présenter les résultats de manière claire, professionnelle et pédagogique

🚀 Fonctionnalités principales
📊 Vue d’ensemble

Indicateurs clés (KPIs) :
cours actuel, plus haut / plus bas, volume moyen, amplitude

Graphique chandelier japonais
avec moyennes mobiles (20 et 50 jours)

Analyse mensuelle :
volumes et performances par mois

📈 Analyse technique

Bandes de Bollinger (zones de surachat / survente)

RSI (Relative Strength Index) avec interprétation visuelle

Analyse de la volatilité sur période glissante

Sélection dynamique de la période d’analyse à partir du 30/01 pour ce fichier mais peut être modifiable  :

1 mois

3 mois

6 mois

1 an

📉 Analyse de la performance

Statistiques globales :

performance annuelle

meilleure et pire journée

Distribution des rendements :
histogramme et boîte à moustaches

Rendements cumulés

Ratios avancés :

ratio de Sharpe

pourcentage de jours positifs

🎯 Indicateurs avancés

Supports et résistances détectés automatiquement

Momentum sur 10 jours

Analyse de tendance par régression linéaire

Analyse des volumes avec moyenne mobile

📋 Données et interactivité

Tableau interactif avec filtrage par date

Statistiques dynamiques selon la période sélectionnée

Export des données au format CSV

Mise en forme professionnelle des tableaux

🛠️ Installation et exécution
▶️ Installation locale

Récupérer les fichiers du projet :

safran_analysis.py

SAFRAN_data_bourse.txt

requirements.txt

Installer les dépendances :

pip install -r requirements.txt


Lancer l’application :


python -m streamlit run safran_analysis.py


L’application s’ouvre automatiquement à l’adresse :
👉 http://localhost:8501

☁️ Déploiement sur Streamlit Cloud

Créer un compte sur Streamlit Cloud

Créer un repository GitHub contenant :

safran_analysis.py

SAFRAN_data_bourse.txt

requirements.txt

Déployer l’application via l’interface Streamlit Cloud

L’application sera accessible via une URL publique.

📁 Structure du projet
safran-analysis/
│
├── safran_analysis.py          # Application Streamlit principale
├── SAFRAN_data_bourse.txt      # Données boursières (TSV)
├── requirements.txt            # Dépendances Python
└── README.md                   # Documentation du projet

🎨 Design et interface

L’interface adopte une thématique aéronautique professionnelle, inspirée de l’identité Safran.

Couleurs :

Rouge Safran #E4002B

Bleu aéronautique #003D7A

Cyan d’accent #00B8D4

Design :

fond sombre

cartes statistiques

animations légères

Typographie : Roboto

📊 Format des données

Les données sont au format TSV avec les colonnes suivantes :

date	ouv	haut	bas	clot	vol	devise


date : DD/MM/YYYY HH:MM

ouv : prix d’ouverture

haut : plus haut

bas : plus bas

clot : prix de clôture

vol : volume échangé

devise : EUR

🔧 Personnalisation

Les couleurs et paramètres visuels sont modifiables en début de fichier :

SAFRAN_RED = "#E4002B"
SAFRAN_BLUE = "#003D7A"
ACCENT_COLOR = "#00B8D4"

⚠️ Avertissement

Ce projet est réalisé à des fins pédagogiques uniquement.
Il ne constitue en aucun cas un conseil en investissement.

📝 Licence

Projet éducatif — utilisation libre pour l’apprentissage et l’analyse personnelle.

Projet réalisé avec Streamlit
Analyse boursière Safran — Année 2025