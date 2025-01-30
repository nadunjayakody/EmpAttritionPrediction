import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import joblib


# Page Configuration
st.set_page_config(page_title='Employee Attrition Dashboard', layout='wide')

class EmployeeAttritionDashboard:
    def __init__(self):
        # Load pre-trained model and preprocessor
        self.model = joblib.load('attrition_model.joblib')
        
        # Initialize session state
        if 'df' not in st.session_state:
            st.session_state.df = None

    def upload_dataset(self):
        """Dataset Upload Tab"""
        st.header('Upload Employee Dataset')
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type=['csv'], 
            help="Upload your employee dataset in CSV format"
        )
        
        if uploaded_file is not None:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Create a processed copy
            df_processed = df.copy()
            
            # Basic data preview
            st.subheader('Dataset Preview')
            st.dataframe(df.head(10))

            # Explicitly convert Attrition to numeric
            if 'Attrition' in df.columns:
                # Convert 'Yes'/'No' to 1/0
                df['Attrition_Numeric'] = df['Attrition'].map({'Yes': 1, 'No': 0})
                
                # If mapping fails, use pandas get_dummies or another method
                if df['Attrition_Numeric'].isnull().any():
                    # Fallback method
                    df['Attrition_Numeric'] = (df['Attrition'] == 'Yes').astype(int)
            
            # Basic dataset info
            col1, col2 = st.columns(2)
            with col1:
                st.metric('Total Rows', df.shape[0])
                st.metric('Total Columns', df.shape[1])
            
            with col2:
                # Use the new numeric column for calculations
                if 'Attrition_Numeric' in df.columns:
                    st.metric('Attrition Rate',
                        f"{df['Attrition_Numeric'].mean():.2%}")
            
            # Save to session state
            # Use the original dataframe, but add the numeric attrition column
            df_processed['Attrition_Numeric'] = df['Attrition_Numeric']
            st.session_state.df = df_processed
            st.success('Dataset Successfully Uploaded!')

    
    def create_visualizations(self):
        """Create Multiple Interactive Visualizations"""
        if st.session_state.df is None:
            st.warning('Please upload a dataset first!')
            return
        
        df = st.session_state.df
        
        # Ensure we have the numeric attrition column
        if 'Attrition_Numeric' not in df.columns:
            st.error('Attrition numeric column is missing. Please re-upload the dataset.')
            return

        #st.header('Interactive Data Visualizations')
        
        # Create tabs for different visualization types
        tab1, tab2, tab3, tab4 = st.tabs([
            'Attrition Overview', 
            'Demographic Insights', 
            'Job-Related Factors', 
            'Salary & Performance'
        ])
        
        with tab1:
            st.subheader('Attrition Overview')
            
            # Interactive Attrition Pie Chart
            col1, col2 = st.columns(2)
            with col1:
                attrition_counts = df['Attrition_Numeric'].value_counts()
                fig_attrition = px.pie(
                    values=attrition_counts.values,
                    names=attrition_counts.index.map({1: 'Left', 0: 'Retained'}),
                    title='Employee Attrition Distribution',
                    hole=0.3
                )
                st.plotly_chart(fig_attrition)
            
            with col2:
                # Safely handle Department attrition rates
                if 'Department' in df.columns:
                    dept_attrition = df.groupby('Department')['Attrition_Numeric'].mean().reset_index()
                    fig_dept_attrition = px.bar(
                        dept_attrition, 
                        x='Department', 
                        y='Attrition_Numeric',
                        title='Attrition Rate by Department',
                        labels={'Attrition_Numeric': 'Attrition Rate'}
                    )
                    st.plotly_chart(fig_dept_attrition)
                else:
                    st.warning("Department column not found in the dataset")
        
        # Similar modifications for other visualization sections
        # Replace 'Attrition' with 'Attrition_Numeric' in other charts
        # For example:
        with tab2:
            st.subheader('Demographic Insights')
            
            # Age Distribution by Attrition
            if 'Age' in df.columns:
                fig_age = px.box(
                    df, 
                    x='Attrition_Numeric', 
                    y='Age',
                    color='Attrition_Numeric',
                    title='Age Distribution by Attrition Status',
                    labels={'Attrition_Numeric': 'Attrition', 'Age': 'Age'}
                )
                st.plotly_chart(fig_age)
            
            # Gender Distribution
            if 'Gender' in df.columns:
                fig_gender = px.histogram(
                    df, 
                    x='Gender', 
                    color='Attrition_Numeric',
                    barmode='group',
                    title='Attrition by Gender',
                    labels={'Attrition_Numeric': 'Attrition Status'}
                )
                st.plotly_chart(fig_gender)

        with tab3:
            st.subheader('Job-Related Factors')
            
            # Job Satisfaction vs Attrition
            if 'JobSatisfaction' in df.columns:
                fig_job_sat = px.histogram(
                    df, 
                    x='JobSatisfaction', 
                    color='Attrition_Numeric',
                    barmode='group',
                    title='Job Satisfaction Impact on Attrition',
                    labels={'JobSatisfaction': 'Job Satisfaction Level', 'Attrition': 'Attrition Status'}
                )
                st.plotly_chart(fig_job_sat)

            # Overtime Impact
            if 'OverTime' in df.columns:
                fig_overtime = px.histogram(
                    df, 
                    x='OverTime', 
                    color='Attrition_Numeric',
                    barmode='group',
                    title='Overtime Impact on Attrition',
                    labels={'OverTime': 'Overtime', 'Attrition': 'Attrition Status'}
                )
                st.plotly_chart(fig_overtime)

        with tab4:
            st.subheader('Salary & Performance Insights')
            
            # Monthly Income vs Years in Current Role
            if 'YearsInCurrentRole' in df.columns and 'MonthlyIncome' in df.columns:
                fig_income_years = px.histogram(
                    df, 
                    x='YearsInCurrentRole', 
                    y='MonthlyIncome',
                    color='Attrition',
                    title='Monthly Income vs Years in Current Role',
                    labels={'YearsInCurrentRole': 'Years in Current Role', 'MonthlyIncome': 'Monthly Income'}
                )
                st.plotly_chart(fig_income_years)

            # Performance Rating vs Attrition
            if 'PerformanceRating' in df.columns:
                fig_perf_rating = px.histogram(
                    df, 
                    x='PerformanceRating', 
                    color='Attrition',
                    barmode='group',
                    title='Performance Rating Distribution by Attrition',
                    labels={'PerformanceRating': 'Performance Rating', 'Attrition': 'Attrition Status'}
                )
                st.plotly_chart(fig_perf_rating)

        

    def prediction_tab(self):
        """Prediction Tab"""
        if st.session_state.df is None:
            st.warning('Please upload a dataset first!')
            return
        
        df = st.session_state.df
        
        st.header(' Attrition Prediction')
        
        # Identify all features except Attrition
        all_features = df.columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Remove Attrition from features
        if 'Attrition' in categorical_features:
            categorical_features.remove('Attrition')
        if 'Attrition_Numeric' in numerical_features:
            numerical_features.remove('Attrition_Numeric')

        categorical_features.remove('Over18')
        numerical_features.remove('EmployeeCount')
        numerical_features.remove('StandardHours')
        
        # Input section
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Numerical Features')
            numerical_inputs = {}
            for feature in numerical_features:
                # Use int for whole number features
                if feature in ['Age', 'YearsAtCompany', 'DistanceFromHome']:
                    numerical_inputs[feature] = st.number_input(
                        f'{feature}', 
                        min_value=0, 
                        value=int(df[feature].median()),
                        step=1
                    )
                else:
                    # Use float for monetary and rate features
                    numerical_inputs[feature] = st.number_input(
                        f'{feature}', 
                        min_value=0.0, 
                        value=float(df[feature].median()),
                        format="%.2f"
                    )
        
        with col2:
            st.subheader('Categorical Features')
            categorical_inputs = {}
            for feature in categorical_features:
                categorical_inputs[feature] = st.selectbox(
                    f'Select {feature}', 
                    options=df[feature].unique()
                )
        
        # Prediction Button
        if st.button('Predict Attrition'):
            # Combine inputs and ensure correct types
            input_data = {}
            for feature, value in numerical_inputs.items():
                input_data[feature] = value
            input_data.update(categorical_inputs)
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            try:
                # Make prediction
                prediction = self.model.predict(input_df)
                prediction_proba = self.model.predict_proba(input_df)

                attrition_probability = prediction_proba[0][1]
                
                # Display results
                if attrition_probability > 0.7:
                    st.error(f'High Attrition Risk (Probability: {prediction_proba[0][1]:.2%})')
                    st.warning('Recommendation: Conduct retention interview')
                else:
                    st.success(f'Low Attrition Risk (Probability: {prediction_proba[0][1]:.2%})')
                    st.info('Employee seems engaged and satisfied')
            
            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")
                st.warning("Ensure all required features are present and correctly formatted")

    def run(self):
        """Main Dashboard Runner"""
        st.title(' Employee Attrition Analytics Dashboard')
        
        # Tabs
        tab1, tab2, tab3 = st.tabs([
            ' Upload Dataset', 
            ' Data Visualization', 
            ' Prediction'
        ])
        
        with tab1:
            self.upload_dataset()
        
        with tab2:
            if st.session_state.df is not None:
                self.create_visualizations()
        
        with tab3:
            self.prediction_tab()

# Run the Dashboard
def main():
    dashboard = EmployeeAttritionDashboard()
    dashboard.run()

if __name__ == '__main__':
    main()