import streamlit as st
import pandas as pd
import pickle
import requests
import base64

@st.cache
def read_model(url):
    response = requests.get(url)
    open("temp.pkl", "wb").write(response.content)
    with open("temp.pkl", "rb") as f:
        svm_classifier = pickle.load(f)
    return svm_classifier


def read_tf(url):
    response = requests.get(url)
    open("temp.pkl", "wb").write(response.content)
    with open("temp.pkl", "rb") as f:
        preprocessing = pickle.load(f)
    return preprocessing

svm_classifier = read_model("https://github.com/manika-lamba/ml/raw/main/model2.pkl")
preprocessing = read_tf("https://github.com/manika-lamba/ml/raw/main/preprocessing.pkl")

# Function to predict the category for a given abstract
def predict_category(abstract):
    # Preprocess the abstract
    abstract_preprocessed = preprocessing.transform([abstract])
    # Make prediction
    prediction = svm_classifier.predict(abstract_preprocessed)
    return prediction

# Create sidebar

# Create tab for choosing CSV file
st.sidebar.header("Choose CSV File with 'Abstract' field")
uploaded_file = st.sidebar.file_uploader("", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')
    st.dataframe(df)
    # Tag the "Abstract" column with the corresponding categories
    df['category'] = df['Abstract'].apply(predict_category)
    st.dataframe(df)

st.sidebar.header("Download Results")
st.sidebar.text("Download the tagged results as a CSV file.")

# Create a download button
if st.sidebar.button("Download"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="results.csv">Download csv file</a>'
    st.markdown(href, unsafe_allow_html=True)


#if st.sidebar.button("Download"):
#    results.to_csv('result.csv', index=False)
#    st.sidebar.file_downloader("Download", 'results.csv')
#    st.sidebar.success('File downloaded')

#if st.sidebar.button("Download"):
#     csv = df['category'].to_csv('results.csv', index=False)
#     st.sidebar.file_downloader("Download", 'results.csv')
#     st.sidebar.success('File downloaded')

#if st.sidebar.button("Download"):
#    csv = df.to_csv(index=False)
#    b64 = base64.b64encode(csv.encode()).decode()
#    href = f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download CSV File</a>'
#    st.markdown(href, unsafe_allow_html=True)

st.title("About")
st.subheader("You can tag your input CSV file of theses and dissertations with Library Science, Archival Studies, and Information Science categories. The screen will show the output.")
