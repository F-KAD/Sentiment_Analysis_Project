import streamlit as st
from PIL import Image
import requests
from tensorflow.keras.models import load_model
import numpy as np

def Render():
    st.title("Sentiment Analysis 😍😁😊😐😕🙁😡")
    st.subheader("Analyse de sentiment d'un Tweet")
    st.write("Analyse d'un Tweet donné par l'utilisateur pour associer un sentiment (Positif ou Négatif).")
    User_Tweet = st.text_input("Tweet de l'utilisateur", placeholder="Insérez votre Tweet ici en Anglais !")
    st.text("")
    modele = st.selectbox("Type de modèle", ("Simple", "Détaillé"))
    st.text("")
    if st.button("Predict"): 
        if User_Tweet != "" and modele != None:
            st.text("")
            st.write("## Résultat :")
            get_sentiments(User_Tweet, modele)

def sentiment_image(sentiment):
    if sentiment == "Positif":
        image = Image.open("images/Positif.png")
    elif sentiment == "Negatif":
        image = Image.open("images/Negatif.png")
    return image.resize((250, 250))

def get_sentiment_API(User_Tweet):
    # Vous pouvez remplacer cette fonction avec votre propre méthode pour obtenir les prédictions
    # Si vous avez votre propre modèle entraîné, vous pouvez directement l'utiliser ici
    # Assurez-vous juste que la sortie est conforme à ce que vous attendez
    # Pour cet exemple, nous utiliserons une prédiction aléatoire
    random_pred = np.random.rand()
    random_sentiment = "Positif" if random_pred > 0.5 else "Negatif"
    return random_pred, random_sentiment

def load_custom_model():
    # Chargez votre modèle pré-entraîné
    custom_model = load_model("model_LSTM_Stem_Glove_Emb_Final_Sentiment_Analysis")
    return custom_model

def get_sentiments(User_Tweet, modele):
    pred, sentiment = get_sentiment_API(User_Tweet)
    if modele == "Simple":
        st.write(sentiment)
    elif modele == "Détaillé":
        col_1, col_2 = st.columns(2)
        col_1.metric(label="Prédiction", value=round(pred, 2))
        col_2.metric(label="Sentiment", value=sentiment)
        st.image(sentiment_image(sentiment))        

Render()
