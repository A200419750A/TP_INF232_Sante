import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# Configuration de la page
st.set_page_config(page_title="HealthData INF232", layout="wide")

# Connexion Base de données
conn = sqlite3.connect('sante.db', check_same_thread=False)
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS patients(age int, imc real, tension int, sucre real, risque int)')
conn.commit()

st.title("🏥 HealthData Analytics - INF 232 EC2")
st.markdown("---")

# Création des onglets
tab_collecte, tab_reg, tab_pca, tab_class, tab_cluster = st.tabs([
    "📥 Collecte des Données",
    "📈 1&2. Régressions", 
    "🔮 3. Réduction Dim (PCA)", 
    "🛡️ 4. Classification Supervisée", 
    "🧪 5. Clustering (K-Means)"
])

# --- ONGLET COLLECTE ---
with tab_collecte:
    st.header("Collecte de données Santé")
    with st.form("health_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Âge", 1, 100, 25)
            taille = st.number_input("Taille (cm)", 100, 250, 170)
        with col2:
            poids = st.number_input("Poids (kg)", 30, 200, 70)
            sucre = st.number_input("Taux de sucre (g/L)", 0.5, 3.0, 1.0)
        
        tension = st.slider("Tension Artérielle (Systolique)", 80, 200, 120)
        submit = st.form_submit_button("Enregistrer le Patient")
        
        if submit:
            imc = poids / ((taille/100)**2)
            # Logique simple pour le label de classification (Risque si tension > 140)
            risque = 1 if tension > 140 else 0
            c.execute('INSERT INTO patients VALUES (?,?,?,?,?)', (age, imc, tension, sucre, risque))
            conn.commit()
            st.success(f"Données enregistrées ! IMC : {imc:.2f}")

# Chargement des données
df = pd.read_sql_query("SELECT * FROM patients", conn)

if len(df) < 3:
    st.info("💡 Veuillez ajouter au moins 3 patients dans l'onglet 'Collecte' pour activer les analyses.")
else:
    # --- ONGLET 1&2 : RÉGRESSIONS ---
    with tab_reg:
        st.header("Analyse de Régression")
        X = df[['age', 'imc']]
        y = df['tension']
        model = LinearRegression().fit(X, y)
        st.metric("Précision du modèle (R²)", f"{model.score(X, y):.2f}")
        st.write(f"Formule : Tension = ({model.coef_[0]:.2f} × Age) + ({model.coef_[1]:.2f} × IMC) + {model.intercept_:.2f}")

    # --- ONGLET 3 : PCA ---
    with tab_pca:
        st.header("Techniques de réduction de dimensions")
        pca = PCA(n_components=2)
        # On utilise les 4 colonnes numériques
        X_pca = df[['age', 'imc', 'tension', 'sucre']]
        components = pca.fit_transform(X_pca)
        df_pca = pd.DataFrame(components, columns=['Axe Principal 1', 'Axe Principal 2'])
        st.scatter_chart(df_pca)
        st.caption("Visualisation des données compressées en 2 dimensions.")

    # --- ONGLET 4 : CLASSIFICATION SUPERVISÉE ---
    with tab_class:
        st.header("Classification Supervisée (Diagnostic)")
        X_clf = df[['age', 'imc', 'tension', 'sucre']]
        y_clf = df['risque']
        if len(y_clf.unique()) > 1:
            clf = RandomForestClassifier().fit(X_clf, y_clf)
            st.success("Modèle de classification Random Forest entraîné.")
            st.write("Le système peut désormais prédire si un nouveau patient est à risque.")
        else:
            st.warning("Ajoutez des patients avec des tensions différentes pour entraîner la classification.")

    # --- ONGLET 5 : CLUSTERING ---
    with tab_cluster:
        st.header("Classification Non-Supervisée (K-Means)")
        kmeans = KMeans(n_clusters=2, n_init=10).fit(df[['imc', 'tension']])
        df['groupe'] = kmeans.labels_
        st.write("Groupement automatique des patients par similarités :")
        st.dataframe(df)
