import flask
from nltk.stem import PorterStemmer
import spacy
import pandas as pd
import mlflow
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

app = flask.Flask(__name__)

#Données
df = pd.read_csv("CSV/df_sample_large.csv")
#spacy    
nlp = spacy.load("en_core_web_sm")
#Tokenizer
tokenizer = Tokenizer(num_words = 10000)
tokenizer.fit_on_texts(df["text"])
#Model
#model = mlflow.keras.load_model("mlruns/2/053fdfca9b8d430bba1e4bd2c9933da9/artifacts/model")
model = mlflow.keras.load_model("mlruns/913197713471476778/238a99e028414a93b737b8c11788c147/artifacts/model")

def tweet_clean(tweet):
    clean = " ".join([PorterStemmer().stem(token.text) for token in nlp(tweet)])
    return clean

def tweet_padded(tweet):
    twt_clean = tweet_clean(tweet)
    padded = pad_sequences(tokenizer.texts_to_sequences([twt_clean]), maxlen = 50, padding = "post", truncating = "post")
    return padded

def tweet_predict(tweet):
    twt_padded = tweet_padded(tweet)
    pred = float(model.predict(twt_padded)[0][0])
    return pred

def tweet_sentiment(pred):
    if pred > 0.5:
        sentiment = "Positif"
    else:
        sentiment = "Negatif"
    return sentiment

@app.route("/", methods = ["GET"])
def index():
    return "Bienvenu sur l'API Twitter Sentiment Analysis"

@app.route("/DL", methods = ["GET"])
def DL():
    tweet = flask.request.args.get("tweet", default = "", type = str)
    pred = tweet_predict(tweet)
    sentiment = tweet_sentiment(pred)    
    return {"Tweet": tweet, "Pred": pred, "Sentiment": sentiment}

if __name__ == "__main__":
    app.run()