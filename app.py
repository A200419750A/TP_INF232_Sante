import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# Configuration
st.set_page_config(page_title="HealthData INF232", layout="wide")
conn = sqlite3.connect('sante.db', check_same_thread=False)
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS patients(age int, imc real, tension int, sucre real, risque int)')
conn.commit()

st.title("🏥 Application de Collecte et d'Analyse - INF 232")

# Création des onglets pour chaque point du cours
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1&2. Régressions", 
    "3. Réduction Dim (PCA)", 
    "4. Classification Supervisée", 
    "5. Clustering (Non-Supervisé)",
    "Collecte des Données"
])

# --- ONGLET COLLECTE (Pour avoir des données) ---
with tab5:
    st.header("Collecte de données Santé")
    with st.form("health_form"):
        age = st.number_input("Âge", 1, 100, 25)
        taille = st.number_input("Taille (cm)", 100, 250, 170)
        poids = st.number_input("Poids (kg)", 30, 200, 70)
        tension = st.slider("Tension Artérielle", 80, 200, 120)
        sucre = st.number_input("Taux de sucre (g/L)", 0.5, 3.0, 1.0)
        submit = st.form_submit_button("Enregistrer")
        
        if submit:
            imc = poids / ((taille/100)**2)
            # On simule un risque pour la classification (1 si tension > 140, sinon 0)
            risque = 1 if tension > 140 or imc > 30 else 0
            c.execute('INSERT INTO patients VALUES (?,?,?,?,?)', (age, imc, tension, sucre, risque))
            conn.commit()
            st.success("Données enregistrées avec succès !")

# Charger les données pour l'analyse
df = pd.read_sql_query("SELECT * FROM patients", conn)

if len(df) < 3:
    st.warning("Veuillez entrer au moins 3 patients dans l'onglet 'Collecte' pour activer les analyses.")
else:
    # --- ONGLET 1&2 : RÉGRESSIONS ---
    with tab1:
        st.header("Régression Linéaire (Simple & Multiple)")
        st.write("Objectif : Prédire la Tension en fonction de l'Âge et de l'IMC.")
        X = df[['age', 'imc']]
        y = df['tension']
        model_reg = LinearRegression().fit(X, y)
        st.metric("Précision du modèle (R²)", f"{model_reg.score(X, y):.2f}")
        st.write(f"Équation : Tension = {model_reg.coef_[0]:.2f}*Age + {model_reg.coef_[1]:.2f}*IMC + {model_reg.intercept_:.2f}")

    # --- ONGLET 3 : PCA ---
    with tab2:
        st.header("Réduction des dimensions (PCA)")
        pca = PCA(n_components=2)
        components = pca.fit_transform(df[['age', 'imc', 'tension', 'sucre']])
        df_pca = pd.DataFrame(components, columns=['Axe 1', 'Axe 2'])
        st.scatter_chart(df_pca)
        st.write("On réduit ici les 4 variables en 2 axes principaux pour visualiser les données.")

    # --- ONGLET 4 : CLASSIFICATION SUPERVISÉE ---
    with tab4:
        st.header("Classification (Random Forest)")
        X_clf = df[['age', 'imc', 'tension', 'sucre']]
        y_clf = df['risque']
        clf = RandomForestClassifier().fit(X_clf, y_clf)
        st.write("Le modèle apprend à classer les patients 'À risque' ou 'Sains'.")
        st.info("Algorithme utilisé : Forêt Aléatoire (Robustesse accrue)")

    # --- ONGLET 5 : CLUSTERING ---
    with
