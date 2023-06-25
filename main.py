import pandas as pd
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
import streamlit as st
import matplotlib.pyplot as plt


moviesData = pd.read_csv('data/train_data.txt', sep=' ::: ', engine='python', names=['Title','Genre','Description'])
# moviesData = pd.read_csv('data/test_data_solution.txt', sep=' ::: ', engine='python', names=['Title','Genre','Description'])

moviesData['Title'] = moviesData['Title'].replace({' \(.*?\)': ''}, regex=True)


def preprocessData():
    target = list(moviesData['Genre'])
    features = list(moviesData['Description'])

    return features, target

def predict(model):
    try:
        os.remove('last_dump.pkl')
    except:
        print('Nothing to delete')

    features, target = preprocessData()
    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.2)
    vectorizer = CountVectorizer()
    features_train_vectorized = vectorizer.fit_transform(features_train)

    if model == 1:
        classifier = LinearSVC()
    if model == 2:
        classifier = LogisticRegression()
    
    classifier.fit(features_train_vectorized, target_train)
    X_test_vectorized = vectorizer.transform(features_test)
    y_pred = classifier.predict(X_test_vectorized)

    with open('last_dump.pkl', 'wb') as f:
        pickle.dump((classifier, vectorizer), f)

    unique_targets = np.unique(target_test)

    cm = confusion_matrix(target_test, y_pred)
    ac = accuracy_score(target_test, y_pred)
    st.write(pd.DataFrame(cm, columns=unique_targets, index=unique_targets))
    st.write(ac)

def predict_from_save(movie_description):
    with open('last_dump.pkl', 'rb') as f:
        classifier, vectorizer = pickle.load(f)
    
    movie_description = [movie_description]
    vectorized_description = vectorizer.transform(movie_description)

    prediction = classifier.predict(vectorized_description) 
    return prediction

def categories():
    genres = []
    for genre in moviesData['Genre']:
        if genre not in genres:
            genres.append(genre)

    counters = []
    for i in range(len(genres)):
        counters.append(moviesData['Genre'].value_counts()[genres[i]])

    fig, ax = plt.subplots()
    ax.bar(genres, counters)

    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    fig.align_xlabels()
    st.pyplot(fig)

def main():
    st.title('Movie category prediction')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        SP = st.button('Singular prediction')
    with col2:
        MC = st.button('Movie categories')
    with col3:
        LinearSVC = st.button('LinearSVC')
    with col4:
        LR = st.button('Logistic regression')
    
    if MC:
        categories()
    elif LinearSVC:
        predict(1)
    elif LR:
        predict(2)
    else:
        with st.form("my_form"):
            st.write("Classification for singular movie from user selected description.")
            movieDesc = st.text_input('Classification by last shown algorithm', placeholder='Enter movie description here')

            submitted = st.form_submit_button("Submit")
            if submitted:
                st.write(predict_from_save(movieDesc))


    
if __name__ == "__main__":
    main()