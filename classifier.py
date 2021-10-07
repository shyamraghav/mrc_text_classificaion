import pickle
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# import keras
# from keras.models import Sequential
# from keras import layers
# import tensorflow
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

class MrCooper():

    def __init__(self):
        # self.nn_model = pickle.load(open("./venv/models/NN_model.pkl",'rb'))
        self.ml_model = pickle.load(open("./venv/models/ML_NB_model.pkl",'rb'))
        self.decoder = pickle.load(open("./venv/models/output_encoder.pkl",'rb'))
        self.vectorizer = pickle.load(open("./venv/models/vectorizer.pkl",'rb'))

    def classify(self, text):
        processed_text = self.pre_process(text)
        # print(processed_text)
        vectorized_text = self.vectorizer.transform(processed_text)
        input_feature = vectorized_text.toarray()
        # print(input_feature)
        # NN_prediction, NN_confidence = self.predict(input_feature,model=NN)
        ML_prediction, ML_confidence = self.predict(input_feature,model= 'ML')

        return {'ML model' : { "Predicted Class" : ML_prediction, "Confidence" : ML_confidence},
                'NN model' : { "Predicted Class" : NN_prediction, "Confidence" : NN_confidence}}

    def predict(self,vector,model):

        if model == 'NN':
            pred,confidence = self.nn_model.predict(vector)
        elif model == 'ML':
            pred = self.ml_model.predict(vector)
            output_class = self.decoder.inverse_transform(pred)
            confidence = max(self.ml_model.predict_proba(vector))
            # print(output_class,max(confidence))
            # print("Till here Successfull")
        return output_class[0],max(confidence)

    def pre_process(self, row):
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(row)
        lemma = [lemmatizer.lemmatize(X) for X in tokens]
        stop = [word for word in lemma if word not in stop_words]
        return [' '.join(stop)]



