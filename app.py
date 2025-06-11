import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
from io import BytesIO
import tempfile
import plotly.express as px
from datetime import datetime

# App title and config
st.set_page_config(page_title="DataStory AI", page_icon="📊", layout="wide")
st.title("📊 DataStory AI")
st.caption("SQL-Free Data Analysis with Google Gemini")

# Sidebar for settings and file upload
with st.sidebar:
    st.header("Configuration")
    
    # Gemini API key input
    if 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = ""
        
    api_key = st.text_input("Google Gemini API Key", 
                           value=st.session_state.gemini_api_key,
                           type="password",
                           help="Get your API key from Google AI Studio")
    
    if api_key:
        st.session_state.gemini_api_key = api_key
        try:
            genai.configure(api_key=api_key)
            st.success("API key configured successfully!")
        except Exception as e:
            st.error(f"Invalid API key: {str(e)}")
    
    st.divider()
    
    # File upload
    uploaded_file = st.file_uploader("Upload Data File", 
                                    type=["csv", "xlsx", "xls"],
                                    help="Upload CSV or Excel files for analysis")
    
    if uploaded_file:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        try:
            if file_ext == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_ext in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
            st.success("File uploaded successfully!")
            
            # Show basic info
            st.write(f"**Rows:** {len(df)}")
            st.write(f"**Columns:** {', '.join(df.columns)}")
            st.write("**Sample data:**")
            st.dataframe(df.head(3), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Main content area
if 'df' not in st.session_state:
    st.info("Please upload a data file to begin analysis", icon="ℹ️")
    st.stop()

df = st.session_state.df

# Tab layout
tab1, tab2, tab3 = st.tabs(["📈 Visual Explorer", "🔍 AI Analysis", "📊 Data Quality"])

with tab1:
    st.header("Interactive Data Explorer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        chart_type = st.selectbox("Chart Type", 
                                 ["Bar", "Line", "Scatter", "Histogram", "Pie"],
                                 index=0)
        
    with col2:
        x_axis = st.selectbox("X-Axis", df.columns, index=0)
        
    y_options = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    color_options = ["None"] + list(df.columns)
    
    if len(y_options) == 0:
        st.warning("No numeric columns found for Y-axis")
    else:
        y_axis = st.selectbox("Y-Axis", y_options, index=0)
        color_col = st.selectbox("Color By", color_options, index=0)
        
        try:
            if chart_type == "Bar":
                fig = px.bar(df, x=x_axis, y=y_axis, color=None if color_col == "None" else color_col)
            elif chart_type == "Line":
                fig = px.line(df, x=x_axis, y=y_axis, color=None if color_col == "None" else color_col)
            elif chart_type == "Scatter":
                fig = px.scatter(df, x=x_axis, y=y_axis, color=None if color_col == "None" else color_col)
            elif chart_type == "Histogram":
                fig = px.histogram(df, x=x_axis, y=y_axis, color=None if color_col == "None" else color_col)
            elif chart_type == "Pie":
                fig = px.pie(df, names=x_axis, values=y_axis)
                
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generating chart: {str(e)}")

with tab2:
    st.header("AI-Powered Analysis")
    
    if not st.session_state.gemini_api_key:
        st.warning("Please enter your Google Gemini API key in the sidebar")
        st.stop()
    
    query = st.text_area("Ask anything about your data", 
                         placeholder="E.g., What are the trends in this data? What anomalies do you see? What's the correlation between X and Y?")
    
    if st.button("Analyze", type="primary") and query:
        with st.spinner("Analyzing data with Gemini..."):
            try:
                # Prepare data sample for Gemini
                data_sample = df.head(10).to_string()
                
                model = genai.GenerativeModel('models/gemini-1.5-flash')
                prompt = f"""Act as a senior data analyst. Analyze this data and answer the user's question.
                
Data Sample (first 10 rows):
{data_sample}

Data Columns: {', '.join(df.columns)}

User Question: {query}

Respond with:
1. Clear answer to the question
2. 3 key insights from the data
3. Recommended visualizations
4. Any data quality concerns
"""
                response = model.generate_content(prompt)
                
                st.subheader("Analysis Results")
                st.markdown(response.text)
                
                # Save to session for later reference
                st.session_state.last_analysis = {
                    "timestamp": datetime.now(),
                    "query": query,
                    "response": response.text
                }
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
    
    if 'last_analysis' in st.session_state:
        st.divider()
        st.subheader("Last Analysis")
        st.write(f"**Query:** {st.session_state.last_analysis['query']}")
        st.write(f"**Time:** {st.session_state.last_analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        st.markdown(st.session_state.last_analysis['response'])

with tab3:
    st.header("Data Quality Report")
    
    quality_metrics = []
    
    # Completeness check
    completeness = (1 - df.isnull().mean()).to_dict()
    quality_metrics.append({
        "Metric": "Completeness",
        "Description": "Percentage of non-null values",
        "Results": completeness
    })
    
    # Uniqueness check
    uniqueness = (df.nunique() / len(df)).to_dict()
    quality_metrics.append({
        "Metric": "Uniqueness",
        "Description": "Ratio of unique values to total rows",
        "Results": uniqueness
    })
    
    # Data type consistency
    dtypes = {col: str(df[col].dtype) for col in df.columns}
    quality_metrics.append({
        "Metric": "Data Types",
        "Description": "Column data types",
        "Results": dtypes
    })
    
    # Display metrics
    for metric in quality_metrics:
        st.subheader(metric["Metric"])
        st.caption(metric["Description"])
        
        if metric["Metric"] in ["Completeness", "Uniqueness"]:
            col1, col2 = st.columns(2)
            
            with col1:
                worst_5 = sorted(metric["Results"].items(), key=lambda x: x[1])[:5]
                st.write("**Areas Needing Attention:**")
                for col, score in worst_5:
                    st.metric(label=col, value=f"{score:.1%}")
            
            with col2:
                best_5 = sorted(metric["Results"].items(), key=lambda x: x[1], reverse=True)[:5]
                st.write("**Best Performing Columns:**")
                for col, score in best_5:
                    st.metric(label=col, value=f"{score:.1%}")
                    
        elif metric["Metric"] == "Data Types":
            st.json(metric["Results"])
    
    st.divider()
    st.subheader("Suggestions for Improvement")
    st.markdown("""
    - **For incomplete columns:** Consider imputation or investigate why data is missing
    - **For low-uniqueness columns:** Check for data entry errors or consider categorization
    - **For inconsistent data types:** Standardize formats (e.g., dates, numeric values)
    """)

# Footer
st.divider()
st.caption("DataStory AI - A SQL-free data analysis tool powered by Google Gemini | Made for your interview")
