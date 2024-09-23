import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from ifc_processor import IFCProcessor
from excel_handler import ExcelHandler
from utils import allowed_file
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="IFC Data Insights Manager", page_icon="assets/favicon.ico", layout="wide")

st.title("IFC Data Insights Manager")

uploaded_file = st.file_uploader("Choose an IFC file", type="ifc")

if uploaded_file is None:
    with open("sample.ifc", "rb") as f:
        uploaded_file = io.BytesIO(f.read())
    st.info("Using sample IFC file. Upload your own file to replace it.")

if uploaded_file is not None:
    if isinstance(uploaded_file, io.BytesIO) or (uploaded_file and allowed_file(uploaded_file.name)):
        st.success("File processed successfully!")
        
        try:
            ifc_processor = IFCProcessor(uploaded_file)
            ifc_processor.extract_properties()
            properties_df = ifc_processor.properties_df
            
            tab1, tab2, tab3, tab4 = st.tabs(["Properties", "Dashboard", "Quality Control", "Machine Learning"])
            
            with tab1:
                try:
                    st.subheader("Properties")
                    logger.info("Generating Properties table")
                    
                    all_properties = [col for col in properties_df.columns if col not in ['Type', 'Name']]
                    selected_properties = st.multiselect(
                        "Select properties to display",
                        options=all_properties,
                        default=all_properties,
                        key="property_select"
                    )
                    
                    filtered_df = properties_df[["Type", "Name"] + selected_properties]
                    
                    logger.info(f"Filtered DataFrame shape: {filtered_df.shape}")
                    logger.info(f"Filtered DataFrame columns: {filtered_df.columns.tolist()}")
                    
                    edited_df = st.data_editor(
                        filtered_df,
                        num_rows="dynamic",
                        use_container_width=True,
                        column_config={
                            "Type": st.column_config.TextColumn(disabled=True),
                            "Name": st.column_config.TextColumn(disabled=True),
                        }
                    )
                    
                    logger.info("Properties table generated successfully")
                    
                    if st.button("Export to Excel"):
                        excel_handler = ExcelHandler()
                        excel_data = excel_handler.export_to_excel(edited_df)
                        st.download_button(
                            label="Download Excel file",
                            data=excel_data,
                            file_name="ifc_properties.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        logger.info("Excel file exported successfully")
                    
                    if st.button("Update IFC"):
                        ifc_processor.update_properties(edited_df)
                        st.success("IFC file updated successfully!")
                        logger.info("IFC file updated successfully")
                except Exception as e:
                    logger.error(f"Error in Properties tab: {str(e)}")
                    st.error(f"An error occurred while generating the properties table: {str(e)}")

            with tab2:
                try:
                    st.subheader("Dashboard")
                    logger.info("Generating Dashboard visualizations")

                    selected_types = st.multiselect("Filter by Type", options=['All'] + sorted(properties_df['Type'].unique().tolist()), default=['All'])
                    selected_properties = st.multiselect("Filter by Properties", options=['All'] + [col for col in properties_df.columns if col not in ['Type', 'Name']], default=['All'])

                    if 'All' not in selected_types:
                        filtered_df = properties_df[properties_df['Type'].isin(selected_types)]
                    else:
                        filtered_df = properties_df

                    if 'All' not in selected_properties:
                        filtered_df = filtered_df[['Type', 'Name'] + selected_properties]

                    st.subheader("Element Type Distribution")
                    type_counts = filtered_df['Type'].value_counts()
                    fig = px.pie(values=type_counts.values, names=type_counts.index, title="Element Types", hole=0.3)
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
                    logger.info("Element Type Distribution chart generated")

                    st.subheader("Property Distribution")
                    selected_property = st.selectbox("Select a property to visualize", options=[col for col in filtered_df.columns if col not in ['Type', 'Name']])
                    if pd.api.types.is_numeric_dtype(filtered_df[selected_property]):
                        fig = px.histogram(filtered_df, x=selected_property, title=f"Distribution of {selected_property}", marginal="box")
                    else:
                        value_counts = filtered_df[selected_property].value_counts()
                        fig = px.bar(x=value_counts.index, y=value_counts.values, title=f"Distribution of {selected_property}")
                        fig.update_traces(texttemplate='%{y}', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                    logger.info(f"Property Distribution chart generated for {selected_property}")

                    st.subheader("Property Correlation")
                    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 1:
                        correlation_matrix = filtered_df[numeric_cols].corr()
                        fig = px.imshow(correlation_matrix, title="Correlation Matrix of Numeric Properties", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
                        fig.update_layout(height=1200, width=1200)
                        st.plotly_chart(fig, use_container_width=True)
                        logger.info("Property Correlation matrix generated")

                        st.subheader("Scatter Plot Matrix")
                        selected_numeric_cols = st.multiselect("Select numeric properties for scatter plot matrix", options=numeric_cols, default=numeric_cols[:4])
                        if len(selected_numeric_cols) > 1:
                            fig = px.scatter_matrix(filtered_df, dimensions=selected_numeric_cols, color="Type")
                            fig.update_layout(height=800)
                            st.plotly_chart(fig, use_container_width=True)
                            logger.info("Scatter Plot Matrix generated")
                        else:
                            st.warning("Please select at least two numeric properties for the scatter plot matrix.")

                        st.subheader("3D Scatter Plot")
                        if len(numeric_cols) >= 3:
                            x_col = st.selectbox("Select X-axis property", options=numeric_cols, key="x_axis")
                            y_col = st.selectbox("Select Y-axis property", options=numeric_cols, key="y_axis")
                            z_col = st.selectbox("Select Z-axis property", options=numeric_cols, key="z_axis")
                            color_col = st.selectbox("Select color property", options=['Type'] + numeric_cols.tolist(), key="color")
                            
                            fig = px.scatter_3d(filtered_df, x=x_col, y=y_col, z=z_col, color=color_col, hover_name="Name", title="3D Scatter Plot")
                            fig.update_layout(scene=dict(xaxis_title=x_col, yaxis_title=y_col, zaxis_title=z_col), height=800)
                            st.plotly_chart(fig, use_container_width=True)
                            logger.info("3D Scatter Plot generated")
                        else:
                            st.warning("Not enough numeric properties for 3D scatter plot.")
                    else:
                        st.write("Not enough numeric properties to create correlation visualizations.")
                        logger.info("Not enough numeric properties for correlation visualizations")
                    
                    logger.info("Dashboard visualizations generated successfully")
                except Exception as e:
                    logger.error(f"Error in Dashboard tab: {str(e)}", exc_info=True)
                    st.error(f"An error occurred while generating the dashboard: {str(e)}")

            with tab3:
                try:
                    st.subheader("Quality Control")
                    logger.info("Generating Quality Control visualizations")
                    
                    ifc_types = ['All'] + sorted(properties_df['Type'].unique().tolist())
                    selected_ifc_types = st.multiselect("Filter by IFC Type", options=ifc_types, default=['All'])
                    
                    if 'All' not in selected_ifc_types:
                        filtered_df = properties_df[properties_df['Type'].isin(selected_ifc_types)]
                    else:
                        filtered_df = properties_df
                    
                    qc_properties = st.multiselect(
                        "Select properties for quality control analysis",
                        options=[col for col in filtered_df.columns if col not in ['Type', 'Name']],
                        default=[col for col in filtered_df.columns if col not in ['Type', 'Name']][:5]
                    )
                    logger.info(f"Selected properties for QC: {qc_properties}")
                    
                    col1, col2 = st.columns(2)
                    
                    for i, prop in enumerate(qc_properties):
                        current_col = col1 if i % 2 == 0 else col2
                        
                        with current_col:
                            st.write(f"### {prop}")
                            
                            empty_percentage = (filtered_df[prop].isnull().sum() / len(filtered_df)) * 100
                            
                            if empty_percentage <= 5:
                                color = "green"
                            elif empty_percentage <= 10:
                                color = "orange"
                            else:
                                color = "red"
                            st.markdown(f"<span style='color:{color}'>Empty Percentage: {empty_percentage:.2f}%</span>", unsafe_allow_html=True)
                            logger.info(f"Empty percentage for {prop}: {empty_percentage:.2f}%")
                            
                            if pd.api.types.is_numeric_dtype(filtered_df[prop]):
                                fig = px.histogram(filtered_df, x=prop, title=f"Distribution of {prop}")
                            else:
                                value_counts = filtered_df[prop].value_counts()
                                fig = px.bar(x=value_counts.index, y=value_counts.values, title=f"Distribution of {prop}")
                            
                            fig.update_layout(height=300, width=400)
                            st.plotly_chart(fig)
                            logger.info(f"Distribution chart generated for {prop}")
                            
                            if not pd.api.types.is_numeric_dtype(filtered_df[prop]):
                                st.write("Value Percentages:")
                                value_counts = filtered_df[prop].value_counts(normalize=True) * 100
                                for val, pct in value_counts.items():
                                    st.write(f"{val}: {pct:.2f}%")
                                logger.info(f"Value percentages displayed for {prop}")
                            
                            st.markdown("---")
                    
                    logger.info("Quality Control visualizations generated successfully")
                except Exception as e:
                    logger.error(f"Error in Quality Control tab: {str(e)}")
                    st.error(f"An error occurred while generating the quality control report: {str(e)}")

            with tab4:
                st.subheader("Machine Learning Analysis")
                
                numeric_cols = properties_df.select_dtypes(include=[np.number]).columns.tolist()
                
                ml_option = st.selectbox("Select Machine Learning Task", ["Clustering", "Regression", "Classification"])
                
                if ml_option == "Clustering":
                    st.subheader("K-means Clustering")
                    selected_features = st.multiselect("Select features for clustering", options=numeric_cols, default=numeric_cols[:3])
                    
                    if len(selected_features) < 2:
                        st.warning("Please select at least two features for clustering.")
                    else:
                        X = properties_df[selected_features]
                        
                        imputer = SimpleImputer(strategy='mean')
                        X_imputed = imputer.fit_transform(X)
                        
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X_imputed)
                        
                        n_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=3)
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        clusters = kmeans.fit_predict(X_scaled)
                        
                        properties_df['Cluster'] = clusters
                        
                        if len(selected_features) == 2:
                            fig = px.scatter(
                                properties_df,
                                x=selected_features[0],
                                y=selected_features[1],
                                color='Cluster',
                                hover_data=['Type', 'Name'],
                                title=f"K-means Clustering ({selected_features[0]} vs {selected_features[1]})"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        elif len(selected_features) >= 3:
                            fig = px.scatter_3d(
                                properties_df,
                                x=selected_features[0],
                                y=selected_features[1],
                                z=selected_features[2],
                                color='Cluster',
                                hover_data=['Type', 'Name'],
                                title=f"K-means Clustering ({selected_features[0]} vs {selected_features[1]} vs {selected_features[2]})"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.subheader("Cluster Statistics")
                        for cluster in range(n_clusters):
                            cluster_data = properties_df[properties_df['Cluster'] == cluster]
                            st.write(f"Cluster {cluster}:")
                            st.write(f"- Number of elements: {len(cluster_data)}")
                            st.write(f"- Most common element types: {cluster_data['Type'].value_counts().head(3).to_dict()}")
                            st.write("- Feature averages:")
                            for feature in selected_features:
                                st.write(f"  - {feature}: {cluster_data[feature].mean():.2f}")
                            st.write("---")
                
                elif ml_option == "Regression":
                    st.subheader("Linear Regression")
                    
                    target_variable = st.selectbox("Select target variable for prediction", options=numeric_cols)
                    
                    feature_cols = st.multiselect("Select features for prediction", options=[col for col in numeric_cols if col != target_variable], default=[col for col in numeric_cols if col != target_variable][:3])
                    
                    if len(feature_cols) > 0 and st.button("Run Linear Regression"):
                        X = properties_df[feature_cols]
                        y = properties_df[target_variable]
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        model = LinearRegression()
                        model.fit(X_train_scaled, y_train)
                        
                        y_pred = model.predict(X_test_scaled)
                        
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        st.write(f"Mean Squared Error: {mse:.4f}")
                        st.write(f"R-squared Score: {r2:.4f}")
                        
                        fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'}, title=f'Actual vs Predicted {target_variable}')
                        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name='Ideal'))
                        st.plotly_chart(fig)
                        
                        st.subheader("Feature Importance")
                        feature_importance = pd.DataFrame({'Feature': feature_cols, 'Importance': model.coef_})
                        feature_importance = feature_importance.sort_values('Importance', ascending=False)
                        fig = px.bar(feature_importance, x='Feature', y='Importance', title='Feature Importance')
                        st.plotly_chart(fig)
                
                elif ml_option == "Classification":
                    st.subheader("Random Forest Classification")
                    
                    categorical_cols = properties_df.select_dtypes(include=['object']).columns.tolist()
                    target_variable = st.selectbox("Select target variable for classification", options=categorical_cols)
                    
                    feature_cols = st.multiselect("Select features for classification", 
                                                  options=[col for col in properties_df.columns if col not in [target_variable, 'Type', 'Name']], 
                                                  default=[col for col in numeric_cols if col != target_variable][:3])
                    
                    if len(feature_cols) > 0 and st.button("Run Random Forest Classification"):
                        X = properties_df[feature_cols]
                        y = properties_df[target_variable]
                        
                        le = LabelEncoder()
                        y_encoded = le.fit_transform(y)
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
                        
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                        rf_model.fit(X_train_scaled, y_train)
                        
                        y_pred = rf_model.predict(X_test_scaled)
                        
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        st.write(f"Accuracy: {accuracy:.4f}")
                        
                        st.subheader("Classification Report")
                        report = classification_report(y_test, y_pred, target_names=le.classes_)
                        st.text(report)
                        
                        st.subheader("Feature Importance")
                        feature_importance = pd.DataFrame({'Feature': feature_cols, 'Importance': rf_model.feature_importances_})
                        feature_importance = feature_importance.sort_values('Importance', ascending=False)
                        fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h', title='Feature Importance')
                        st.plotly_chart(fig)

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            st.error(f"An unexpected error occurred: {str(e)}")

    else:
        st.error("Invalid file type. Please upload an IFC file.")
else:
    st.info("Please upload an IFC file to begin.")