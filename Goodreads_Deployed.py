import streamlit as st
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
model=pickle.load(open('goodreads.pkl','rb'))
tfidfVectorizer = pickle.load(open('tfidf.pkl', 'rb'))

def main():
    st.header('Spoiler Classification')
    st.subheader('Find out whether a given Book/Movie review has a spoiler or not!')
    review = st.text_input('Enter Review', "")
    corpus = []
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    X = tfidfVectorizer.transform(corpus).toarray()
    if st.button("Predict"):
        output=model.predict(X)
        if output==0:
            st.success('The review has No Spoiler')
            st.balloons()
        else:
            st.error('The review has a Spoiler')
    
    st.subheader('Developed by: Maaz Ansari')
if __name__=='__main__':
     main()
