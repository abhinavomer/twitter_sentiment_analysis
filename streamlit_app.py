import pickle
import streamlit as st
st.title("Twitter Sentiment Analysis")
st.image('tsa.jpg')
a=[]
aa=st.text_input("Enter a Tweet:-")
a.append(aa)
if st.button("Predict"):
    vectorizer=pickle.load(open('vectorizer_twitter.pickle','rb'))
    b=vectorizer.transform(a)
    model=pickle.load(open('model_twitter.pickle','rb'))
    pred=model.predict(b)
    if pred==0:
        st.write("Negative")
    else:
        st.write("Positive")