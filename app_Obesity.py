
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load model and encoders
with open('model_Obesity.pkl', 'rb') as file:
    # Load the data from the file
    model, Gender_encoder, family_history_with_overweight_encoder, FAVC_encoder, CAEC_encoder, SMOKE_encoder, SCC_encoder, CALC_encoder, MTRANS_encoder, NObeyesdad_encoder = pickle.load(file)

# Load your DataFrame
# Replace 'your_data.csv' with the actual file name or URL
df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic_raw.csv')
df = df.drop('Height', axis=1)
df = df.drop('Weight', axis=1)

# Streamlit App
st.title('Obesity Level App')

# Define a session state to remember tab selections
if 'tab_selected' not in st.session_state:
    st.session_state.tab_selected = 0

# Create tabs for prediction and visualization
tabs = ['Predict Obesity Level', 'Visualize Data', 'Predict from CSV']
selected_tab = st.radio('Select Tab:', tabs, index=st.session_state.tab_selected)

# Tab selection logic
if selected_tab != st.session_state.tab_selected:
    st.session_state.tab_selected = tabs.index(selected_tab)

# Tab 1: Predict KPIs
if st.session_state.tab_selected == 0:
    st.header('Obesity Level')

    # User Input Form
    Gender = st.selectbox('Gender', Gender_encoder.classes_)
    Age = st.slider('Age', 18, 60, 30)
    family_history_with_overweight = st.selectbox('family_history_with_overweight', family_history_with_overweight_encoder.classes_)
    FAVC = st.selectbox('FAVC', FAVC_encoder.classes_)
    FCVC = st.slider('FCVC', 1, 10, 1)
    NCP = st.slider('NCP', 1, 10, 1)
    CAEC = st.selectbox('CAEC', CAEC_encoder.classes_)
    SMOKE = st.selectbox('SMOKE', SMOKE_encoder.classes_)
    CH2O = st.slider('CH2O', 1, 10, 1)
    SCC = st.selectbox('SCC', SCC_encoder.classes_)
    FAF = st.slider('FAF', 1, 10, 1)
    TUE = st.slider('TUE', 1, 10, 1)
    CALC = st.selectbox('CALC', CALC_encoder.classes_)
    MTRANS = st.selectbox('MTRANS', SMOKE_encoder.classes_)

    # Create a DataFrame for the user input
    user_input = pd.DataFrame({
        'Gender': [Gender],
        'Age': [Age],
        'family_history_with_overweight': [family_history_with_overweight],
        'FAVC': [FAVC],
        'FCVC': [FCVC],
        'NCP': [NCP],
        'CAEC': [CAEC],
        'SMOKE': [SMOKE],
        'CH2O': [CH2O],
        'SCC': [SCC],
        'FAF': [FAF],
        'TUE': [TUE],
        'CALC': [CALC],
        'MTRANS': [MTRANS]
    })

    # Categorical Data Encoding
    user_input['Gender'] = Gender_encoder.transform(user_input['Gender'])
    user_input['family_history_with_overweight'] = family_history_with_overweight_encoder.transform(user_input['family_history_with_overweight'])
    user_input['FAVC'] = FAVC_encoder.transform(user_input['FAVC'])
    user_input['CAEC'] = CAEC_encoder.transform(user_input['CAEC'])
    user_input['SMOKE'] = SMOKE_encoder.transform(user_input['SMOKE'])
    user_input['SCC'] = SCC_encoder.transform(user_input['SCC'])
    user_input['CALC'] = CALC_encoder.transform(user_input['CALC'])
    user_input['MTRANS'] = MTRANS_encoder.transform(user_input['MTRANS'])

    # Predicting
    prediction = model.predict(user_input)

    # Display Result
    st.subheader('Prediction Result:')
    st.write('Obesity Level:', prediction[0])

# Tab 2: Visualize Data
elif st.session_state.tab_selected == 1:
    st.header('Visualize Data')

    # Select condition feature
    condition_feature = st.selectbox('Select Condition Feature:', df.columns)

    # Set default condition values
    default_condition_values = ['Select All'] + df[condition_feature].unique().tolist()

    # Select condition values
    condition_values = st.multiselect('Select Condition Values:', default_condition_values)

    # Handle 'Select All' choice
    if 'Select All' in condition_values:
        condition_values = df[condition_feature].unique().tolist()

    if len(condition_values) > 0:
        # Filter DataFrame based on selected condition
        filtered_df = df[df[condition_feature].isin(condition_values)]

        # Plot the number of employees based on KPIs
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.countplot(x=condition_feature, hue='Obesity Level:', data=filtered_df, palette='viridis')
        plt.title('Number of Obesity Level')
        plt.xlabel(condition_feature)
        plt.ylabel('Number of Person')
        st.pyplot(fig)

# Tab 3: Predict from CSV
elif st.session_state.tab_selected == 2:
    st.header('Predict from CSV')

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    # uploaded_file

    if uploaded_file is not None:
        # Read CSV file
        csv_df_org = pd.read_csv(uploaded_file)
        csv_df_org = csv_df_org.dropna()
        # csv_df_org.columns

        csv_df = csv_df_org.copy()
        csv_df = csv_df.drop('Height', axis=1)
        csv_df = csv_df.drop('Weight', axis=1)



         # Categorical Data Encoding
        csv_df['Gender'] = Gender_encoder.transform(csv_df['Gender'])
        csv_df['family_history_with_overweight'] = family_history_with_overweight_encoder.transform(csv_df['family_history_with_overweight'])
        csv_df['FAVC'] = FAVC_encoder.transform(csv_df['FAVC'])
        csv_df['CAEC'] = CAEC_encoder.transform(csv_df['CAEC'])
        csv_df['SMOKE'] = SMOKE_encoder.transform(csv_df['SMOKE'])
        csv_df['SCC'] = SCC_encoder.transform(csv_df['SCC'])
        csv_df['CALC'] = CALC_encoder.transform(csv_df['CALC'])
        csv_df['MTRANS'] = MTRANS_encoder.transform(csv_df['MTRANS'])


        # Predicting
        predictions = model.predict(csv_df)

        # Add predictions to the DataFrame
        csv_df_org['Obesity Level'] = predictions

        # Display the DataFrame with predictions
        st.subheader('Predicted Results:')
        st.write(csv_df_org)

        # Visualize predictions based on a selected feature
        st.subheader('Visualize Predictions')

        # Select feature for visualization
        feature_for_visualization = st.selectbox('Select Feature for Visualization:', csv_df_org.columns)

        # Plot the number of employees based on KPIs for the selected feature
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.countplot(x=feature_for_visualization, hue='Obesity Level', data=csv_df_org, palette='viridis')
        plt.title(f'Number of Obesity Level - {feature_for_visualization}')
        plt.xlabel(feature_for_visualization)
        plt.ylabel('Number of Person')
        st.pyplot(fig)

