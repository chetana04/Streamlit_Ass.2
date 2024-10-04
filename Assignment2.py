import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_california_housing, load_iris
from mlxtend.frequent_patterns import apriori, association_rules
import streamlit as st
import numpy as np

# Custom CSS for better visuals
st.markdown("""
    <style>
        /* Slider styling */
        .stSlider > div > div > div > div {
            background-color: purple !important;  /* Purple slider color */
        }

        /* Style output text */
        .output-text {
            font-size: 1.5em;
            font-weight: bold;
            color: #4CAF50;
            margin-top: 10px;  /* Added some space above output text */
        }
        
        /* General dropdown arrow styling */
        .css-1wy0on6 { 
            color: #4CAF50 !important;  /* Change the dropdown arrow color */
        }

        /* Selected item text in the selectbox */
        .css-qrbaxs {  
            color: lightblue !important;  /* Light blue font color for selected item */
        }

        /* Dropdown menu items styling */
        .css-26l3qy-menu {
            color: lightblue !important;  /* Light blue font color for dropdown items */
        }

    </style>
""", unsafe_allow_html=True)


# Save models to pickle files
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# Load models from pickle files
@st.cache_resource
def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Load datasets and save models
@st.cache_resource
def load_models():
    boston = fetch_california_housing()
    iris = load_iris()

    # Linear Regression (California Housing)
    X, y = pd.DataFrame(boston.data, columns=boston.feature_names), boston.target
    lin_reg_model = LinearRegression().fit(X, y)
    save_model(lin_reg_model, 'linear_reg_model.pkl')

    # Logistic Regression (Iris)
    X_iris, y_iris = iris.data, iris.target
    log_reg_model = LogisticRegression(max_iter=200).fit(X_iris, y_iris)
    save_model(log_reg_model, 'logistic_reg_model.pkl')

    # Naive Bayes (Iris)
    naive_bayes_model = GaussianNB().fit(X_iris, y_iris)
    save_model(naive_bayes_model, 'naive_bayes_model.pkl')

    # Decision Tree (Iris)
    decision_tree_model = DecisionTreeClassifier().fit(X_iris, y_iris)
    save_model(decision_tree_model, 'decision_tree_model.pkl')

    return iris  # Return iris dataset for later use

iris = load_models()

# Apriori Transactions
# Apriori Transactions with actual product names
transactions = pd.DataFrame({
    'Apple': [1, 0, 1, 0, 1],
    'Guava': [1, 1, 0, 1, 0],
    'Knife': [0, 1, 0, 0, 1],
    'Grapes': [0, 1, 1, 0, 1],
    'Dragonfruit': [1, 0, 0, 1, 0]
})

# Sidebar for user input based on model selection
def get_user_input(model_type):
    if model_type == 'Linear Regression':
        st.sidebar.write("### California Housing Features")
        MedInc = st.sidebar.slider('Median Income (in $1000s)', 0.0, 15.0, 3.0)
        HouseAge = st.sidebar.slider('House Age', 1.0, 52.0, 20.0)
        AveRooms = st.sidebar.slider('Average Rooms', 1.0, 20.0, 6.0)
        AveBedrms = st.sidebar.slider('Average Bedrooms', 0.5, 5.0, 1.0)
        Population = st.sidebar.slider('Population', 1.0, 35682.0, 1000.0)
        AveOccup = st.sidebar.slider('Average Occupants per Household', 0.5, 10.0, 3.0)
        Latitude = st.sidebar.slider('Latitude', 32.0, 42.0, 35.0)
        Longitude = st.sidebar.slider('Longitude', -125.0, -114.0, -120.0)
        return np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])

    elif model_type in ['Logistic Regression', 'Naive Bayes', 'Decision Tree']:
        st.sidebar.write("### Iris Features")
        sepal_length = st.sidebar.slider('Sepal Length', 4.0, 8.0, 5.0)
        sepal_width = st.sidebar.slider('Sepal Width', 2.0, 5.0, 3.0)
        petal_length = st.sidebar.slider('Petal Length', 1.0, 7.0, 4.0)
        petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 1.0)
        return np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    elif model_type == 'Apriori':
        st.sidebar.write("### Select Product Transactions")
        apple = st.sidebar.selectbox('Apple', [1, 0])
        guava = st.sidebar.selectbox('Guava', [1, 0])
        knife = st.sidebar.selectbox('Knife', [1, 0])
        grapes = st.sidebar.selectbox('Grapes', [1, 0])
        dragonfruit = st.sidebar.selectbox('Dragonfruit', [1, 0])
        
        # Return DataFrame with product names
        return pd.DataFrame({
            'Apple': [apple], 
            'Guava': [guava], 
            'Knife': [knife], 
            'Grapes': [grapes], 
            'Dragonfruit': [dragonfruit]
        })

col1, col2 = st.columns([1, 3])

with col1:
    st.image("https://th.bing.com/th?id=OIP.bAla57YthoWCIf-RoXX4qAHaHa&w=250&h=250&c=8&rs=1&qlt=90&o=6&dpr=1.3&pid=3.1&rm=2", width=150)

with col2:
    def h2(text, key=None, color='black'):
        st.markdown(f'<h1 style="font-size: 3em; color: {color};">{text}</h1>', unsafe_allow_html=True)
    h2("Machine Learning Model App", key="heading", color='Purple')

    
# Function to load the model and make predictions
def load_model_and_predict():
    model_choice = st.sidebar.selectbox('Select Model', ['Linear Regression', 'Logistic Regression', 
                                                         'Naive Bayes', 'Apriori', 'Decision Tree'])

    if model_choice == 'Linear Regression':
        lin_reg_model = load_model('linear_reg_model.pkl')
        user_input = get_user_input('Linear Regression')
        prediction = lin_reg_model.predict(user_input)
        st.write(f"### House Price Prediction: {prediction[0]:.2f}")

    elif model_choice == 'Logistic Regression':
        log_reg_model = load_model('logistic_reg_model.pkl')
        user_input = get_user_input('Logistic Regression')
        prediction = log_reg_model.predict(user_input)
        st.write(f"### Iris Classification Prediction: {iris.target_names[prediction][0]}")

    elif model_choice == 'Naive Bayes':
        naive_bayes_model = load_model('naive_bayes_model.pkl')
        user_input = get_user_input('Naive Bayes')
        prediction = naive_bayes_model.predict(user_input)
        st.write(f"### Naive Bayes Prediction: {iris.target_names[prediction][0]}")

    elif model_choice == 'Apriori':
        user_input = get_user_input('Apriori')
        st.write("### Your Transaction Input:")
        st.write(user_input)

        global transactions
        transactions = pd.concat([transactions, user_input], ignore_index=True).fillna(0).astype(int)
        
        frequent_itemsets = apriori(transactions, min_support=0.5, use_colnames=True)
        if frequent_itemsets.empty:
            st.write("No frequent itemsets found with the current minimum support.")
        else:
            st.write("### Frequent Itemsets:")
            st.write(frequent_itemsets)

            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
            st.write("### Association Rules:")
            st.write(rules)

    elif model_choice == 'Decision Tree':
        dec_tree_model = load_model('decision_tree_model.pkl')
        user_input = get_user_input('Decision Tree')
        prediction = dec_tree_model.predict(user_input)
        st.write(f"### Iris Classification Prediction: {iris.target_names[prediction][0]}")

# Call the function to run the app
load_model_and_predict()