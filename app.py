#!/usr/bin/env python
# coding: utf-8
##
# In[1]:


import joblib


# Load the model using a relative path
model = joblib.load("C:/Users/anike/OneDrive/Desktop/Projects/Machine Learning/churn/model.sav")



# In[2]:


# Check if the model has the attribute 'feature_names_in_'
if hasattr(model, 'feature_names_in_'):
    # Get the list of feature names
    feature_names = model.feature_names_in_
    print("Feature names used during training:", feature_names)
else:
    print("This model does not have the attribute 'feature_names_in_'")


# In[3]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Define the preprocess function
def preprocess(df, option, feature_names):
    """
    This function covers all the preprocessing steps on the churn dataframe. It involves selecting important features, encoding categorical data, handling missing values, feature scaling, and splitting the data.
    """
    # Define the binary_map function
    def binary_map(feature):
        return feature.map({'Yes': 1, 'No': 0})

    # Encode binary categorical features
    binary_list = ['SeniorCitizen', 'Dependents', 'PhoneService', 'PaperlessBilling']
    df[binary_list] = df[binary_list].apply(binary_map)

    # Drop values based on operational options
    if option == "Online":
        # Ensure all necessary columns are present
        columns = feature_names
        # Encoding the other categorical features with more than two categories
        df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)
    elif option == "Batch":
        # Ensure all necessary columns are present
        columns = ['Dependents', 'tenure', 'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
                   'MultipleLines_No_phone_service', 'MultipleLines_Yes', 'InternetService_Fiber_optic',
                   'InternetService_No', 'OnlineSecurity_No_internet_service', 'OnlineSecurity_Yes',
                   'OnlineBackup_No_internet_service', 'OnlineBackup_Yes', 'TechSupport_No_internet_service',
                   'TechSupport_Yes', 'StreamingTV_No_internet_service', 'StreamingTV_Yes',
                   'StreamingMovies_No_internet_service', 'StreamingMovies_Yes', 'Contract_One_year',
                   'Contract_Two_year', 'PaymentMethod_Credit_card__automatic_', 'PaymentMethod_Electronic_check']
        # Encoding the other categorical features with more than two categories
        df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)
    else:
        print("Incorrect operational options")

    # Feature scaling
    sc = MinMaxScaler()
    df['tenure'] = sc.fit_transform(df[['tenure']])
    df['MonthlyCharges'] = sc.fit_transform(df[['MonthlyCharges']])
    df['TotalCharges'] = sc.fit_transform(df[['TotalCharges']])
    return df

# Assuming 'model' is your trained scikit-learn model
# Check if the model has the attribute 'feature_names_in_'
if hasattr(model, 'feature_names_in_'):
    # Get the list of feature names
    feature_names = model.feature_names_in_
    print("Feature names used during training:", feature_names)
else:
    print("This model does not have the attribute 'feature_names_in_'")

# Define input data manually
features_dict = {
    'SeniorCitizen': 'No',
    'Dependents': 'No',
    'tenure': 24,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'TechSupport': 'Yes',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'No',
    'Contract': 'Two year',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Credit card (automatic)',
    'MonthlyCharges': 85.0,
    'TotalCharges': 2000.0,
}

# Convert the input data into a DataFrame
features_df = pd.DataFrame([features_dict])

# Preprocess inputs
preprocess_df = preprocess(features_df, 'Online', feature_names)

# Make prediction
prediction = model.predict(preprocess_df)


# In[10]:


import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set custom page configuration
st.set_page_config(
    page_title="Telco Churn Prediction App",
    page_icon=":telephone_receiver:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styles
st.markdown(
    """
    <style>
    .title {
        color: #0066cc;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background-color: #f0f0f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to generate visualizations based on user input
def generate_visualizations(data):
    # Countplot for Churn Distribution
    st.subheader("Churn Distribution")
    fig = px.histogram(data, x='Churn', color='Churn', title='Churn Distribution')
    st.plotly_chart(fig)

    # Pairplot for feature analysis
    st.subheader("Contract")
    
    fig = px.scatter_matrix(data, dimensions=data.columns, color='Churn', title='Pairplot for Feature Analysis')
    st.plotly_chart(fig)

    # Correlation heatmap
    st.subheader("Paperless Billing")
    corr = data.corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.index, y=corr.columns, colorscale='Viridis'))
    st.plotly_chart(fig)
 # Prediction button
    if st.button('Predict'):
        # Generate visualizations based on user input
        generate_visualizations(data)

def generate_visualizations(data):
    st.subheader("Overall Visualization")
    fig = px.histogram(data, x='Churn', color='Churn', title='Overall Churn Distribution')
    st.plotly_chart(fig)

def main():
    # Setting Application title
    st.title('Telco Churn Prediction App')

    # Input data
    data = pd.read_csv('C:/Users/anike/OneDrive/Desktop/Projects/Machine Learning/churn/churn.csv')

    # Display dataset
    st.write("Dataset Overview:")
    st.write(data.head())

    # Input form
    st.subheader("Enter Customer Information:")
    input_data = {}

    # Get user input
    input_data['Churn'] = st.radio("Churn?", options=['Yes', 'No'])

    # Prediction button
    if st.button('Predict'):
        # Generate visualizations based on on user input
        generate_visualizations(data)

def main():
    # Setting Application title
    st.title('Telco Churn Prediction App')

    # Input data
    data = pd.read_csv('your_dataset.csv')

    # Display dataset
    st.write("Dataset Overview:")
    st.write(data.head())

    # Input form
    st.subheader("Enter Customer Information:")
    input_data = {}

    # Get user input
    input_data['Churn'] = st.radio("Churn?", options=['Yes', 'No'])

    # Prediction button
    if st.button('Predict'):
        # Generate visualizations based on user input
        generate_visualizations(data, input_data)

def main():
    # Setting Application title
    st.title('Welcome to the Telco Churn Prediction App')
    
    # Setting Application description
    st.markdown("""
    :dart:  This application predicts customer churn in a
    fictional telecommunications scenario. 
    It offers functionality for both real-time and batch prediction.
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)
    
    # Setting Application sidebar default
    st.sidebar.title('Prediction Options')
    st.sidebar.info('Prediction of Customer Churn')

    # Setting Application sidebar default
    image = Image.open('C:/Users/anike/OneDrive/Desktop/Projects/Machine Learning/churn/App.jpg')
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Customer Churn')
    st.sidebar.image(image)

    # Assuming 'model' is your trained scikit-learn model
    # Check if the model has the attribute 'feature_names_in_'
    if hasattr(model, 'feature_names_in_'):
        # Get the list of feature names
        feature_names = model.feature_names_in_
        print("Feature names used during training:", feature_names)
    else:
        print("This model does not have the attribute 'feature_names_in_'")

    if add_selectbox == "Online":
        st.info("Input data below")
        # Based on our optimal features selection
        st.subheader("Demographic data")
        seniorcitizen = st.selectbox('Senior Citizen:', ('Yes', 'No'))
        dependents = st.selectbox('Dependent:', ('Yes', 'No'))
        st.subheader("Payment data")
        tenure = st.slider('Number of months the customer has stayed with the company', min_value=0, max_value=72, value=0)
        contract = st.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
        paperlessbilling = st.selectbox('Paperless Billing', ('Yes', 'No'))
        PaymentMethod = st.selectbox('PaymentMethod',('Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'))
        monthlycharges = st.number_input('The amount charged to the customer monthly', min_value=0, max_value=150, value=0)
        totalcharges = st.number_input('The total amount charged to the customer',min_value=0, max_value=10000, value=0)

        st.subheader("Services signed up for")
        mutliplelines = st.selectbox("Does the customer have multiple lines",('Yes','No','No phone service'))
        phoneservice = st.selectbox('Phone Service:', ('Yes', 'No'))
        internetservice = st.selectbox("Does the customer have internet service", ('DSL', 'Fiber optic', 'No'))
        onlinesecurity = st.selectbox("Does the customer have online security",('Yes','No','No internet service'))
        onlinebackup = st.selectbox("Does the customer have online backup",('Yes','No','No internet service'))
        techsupport = st.selectbox("Does the customer have technology support", ('Yes','No','No internet service'))
        streamingtv = st.selectbox("Does the customer stream TV", ('Yes','No','No internet service'))
        streamingmovies = st.selectbox("Does the customer stream movies", ('Yes','No','No internet service'))

        data = {
            'SeniorCitizen': seniorcitizen,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phoneservice,
            'MultipleLines': mutliplelines,
            'InternetService': internetservice,
            'OnlineSecurity': onlinesecurity,
            'OnlineBackup': onlinebackup,
            'TechSupport': techsupport,
            'StreamingTV': streamingtv,
            'StreamingMovies': streamingmovies,
            'Contract': contract,
            'PaperlessBilling': paperlessbilling,
            'PaymentMethod': PaymentMethod,
            'MonthlyCharges': monthlycharges,
            'TotalCharges': totalcharges
        }
        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)
        # Preprocess inputs
        preprocess_df = preprocess(features_df, 'Online', feature_names)

        prediction = model.predict(preprocess_df)

        if st.button('Predict'):
            if prediction == 1:
                st.warning('Yes, the customer will terminate the service.')
            else:
                st.success('No, the customer is happy with Telco Services.')

    else:
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            # Get overview of data
            st.write(data.head())
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            # Preprocess inputs
            preprocess_df = preprocess(data, "Batch", feature_names)
            if st.button('Predict'):
                # Get batch prediction
                prediction = model.predict(preprocess_df)
                prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
                prediction_df = prediction_df.replace({1: 'Yes, the customer will terminate the service.',
                                                       0: 'No, the customer is happy with Telco Services.'})

                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction')
                st.write(prediction_df)
                # Plotting
                # Plotting
                st.write('### Visualization')

                # Countplot for churn distribution
                st.subheader("Churn Distribution")
                fig = px.histogram(data, x='Churn', color='Churn', title='Churn Distribution')
                st.plotly_chart(fig)

                # Pairplot for feature analysis
                st.subheader("Pairplot for Feature Analysis")
                fig = px.scatter_matrix(data, dimensions=data.columns, color='Churn', title='Pairplot for Feature Analysis')
                st.plotly_chart(fig)

                # Correlation heatmap
                st.subheader("Correlation Heatmap")
                corr = data.corr()
                fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.index, y=corr.columns, colorscale='Viridis'))
                st.plotly_chart(fig)
                
                # Churn vs Non-Churn Visualizations
                churn_data = data[data['Churn'] == 'Yes']
                non_churn_data = data[data['Churn'] == 'No']

                # Countplot for Churn Distribution
                st.subheader("Churn Distribution")
                fig = px.histogram(data, x='Churn', color='Churn', title='Churn Distribution')
                st.plotly_chart(fig)

                # Pairplot for feature analysis - Churn
                st.subheader("Pairplot for Feature Analysis - Churn")
                fig = px.scatter_matrix(churn_data, dimensions=churn_data.columns, color='Churn', title='Pairplot for Feature Analysis - Churn')
                st.plotly_chart(fig)

                # Pairplot for feature analysis - Non Churn
                st.subheader("Pairplot for Feature Analysis - Non Churn")
                fig = px.scatter_matrix(non_churn_data, dimensions=non_churn_data.columns, color='Churn', title='Pairplot for Feature Analysis - Non Churn')
                st.plotly_chart(fig)

                # Correlation heatmap - Churn
                st.subheader("Correlation Heatmap - Churn")
                corr_churn = churn_data.corr()
                fig = go.Figure(data=go.Heatmap(z=corr_churn.values, x=corr_churn.index, y=corr_churn.columns, colorscale='Viridis'))
                st.plotly_chart(fig)

                # Correlation heatmap - Non Churn
                st.subheader("Correlation Heatmap - Non Churn")
                corr_non_churn = non_churn_data.corr()
                fig = go.Figure(data=go.Heatmap(z=corr_non_churn.values, x=corr_non_churn.index, y=corr_non_churn.columns, colorscale='Viridis'))
                st.plotly_chart(fig)


if __name__ == '__main__':
    main()

