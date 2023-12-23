import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.image("fakenewsheader.png")

model1 = joblib.load('C:\\Users\\VIGNESH\\log_reg.joblib') 
model2 = joblib.load("C:\\Users\\VIGNESH\\Desktop\\random_forest.joblib") 
vectorizer1=joblib.load("C:\\Users\\VIGNESH\\vect.joblib")
vectorizer2=joblib.load("C:\\Users\\VIGNESH\\Desktop\\CountVectorizer.joblib")


st.title("Fake News Detection")

input_news=st.text_area("Enter the News:","")

select_model = st.selectbox("Select a Model", ["Logistic Regression", "Random Forest"])

if st.button("Check"):
    if input_news:

        if select_model == "Logistic Regression":
            user_input_news = vectorizer1.transform([input_news])
            prediction = model1.predict(user_input_news)
        else:
            user_input_news = vectorizer2.transform([input_news])
            prediction = model2.predict(user_input_news)

        # Display the prediction result
        if prediction[0] == 1:
            st.error(f"Fake News Detected! ({select_model})")
        else:
            st.success(f"No Fake News Detected. ({select_model})")
    else:
        st.warning("Please enter a news article.")