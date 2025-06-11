import streamlit as st
import pandas as pd
import google.generativeai as genai
import sqlite3
import tempfile
import os
import plotly.express as px
from datetime import datetime
from pandasql import sqldf

# Configure App
st.set_page_config(
    page_title="DataStory Pro",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 DataStory Pro")
st.caption("Natural Language → SQL Analysis with Gemini AI")

# Initialize Session State
if 'df' not in st.session_state:
    st.session_state.df = None
if 'db_conn' not in st.session_state:
    st.session_state.db_conn = None
if 'active_table' not in st.session_state:
    st.session_state.active_table = None
if 'table_info' not in st.session_state:
    st.session_state.table_info = None
if 'result_df' not in st.session_state:
    st.session_state.result_df = None

# Gemini Configuration
generation_config = {
    "temperature": 0.3,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
]

# ----------------------------------
# Sidebar Configuration
# ----------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Gemini API Key
    if 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = ""
    
    api_key = st.text_input(
        "Google Gemini API Key",
        value=st.session_state.gemini_api_key,
        type="password",
        help="Get from Google AI Studio"
    )
    
    if api_key:
        st.session_state.gemini_api_key = api_key
        try:
            genai.configure(api_key=api_key)
            st.success("✅ API Key Valid")
        except Exception as e:
            st.error(f"Invalid API Key: {str(e)}")

    st.divider()

    # File Upload
    uploaded_file = st.file_uploader(
        "📂 Upload Data File",
        type=["csv", "xlsx", "xls", "db"],
        help="CSV, Excel, or SQLite Database"
    )

    if uploaded_file:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        try:
            if file_ext == 'csv':
                st.session_state.df = pd.read_csv(uploaded_file)
                st.session_state.db_conn = None
                st.success("CSV Loaded")

            elif file_ext in ['xlsx', 'xls']:
                st.session_state.df = pd.read_excel(uploaded_file)
                st.session_state.db_conn = None
                st.success("Excel Loaded")

            elif file_ext == 'db':
                # Save DB to temp file
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                # Connect to SQLite
                conn = sqlite3.connect(tmp_path)
                st.session_state.db_conn = conn

                # Get first table
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()

                if tables:
                    st.session_state.active_table = tables[0][0]
                    st.session_state.df = pd.read_sql(
                        f"SELECT * FROM {st.session_state.active_table} LIMIT 1000",
                        conn
                    )

                    # Get table schema
                    cursor.execute(f"PRAGMA table_info({st.session_state.active_table})")
                    st.session_state.table_info = cursor.fetchall()
                    st.success(f"SQLite Loaded: {st.session_state.active_table}")

                os.unlink(tmp_path)

            # Show sample data
            if st.session_state.df is not None:
                st.write("Data Preview:")
                st.dataframe(st.session_state.df.head(3), use_container_width=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")

    st.divider()
    st.caption("🛠️ Tip: For SQLite, we'll use the first available table")

# ---------------------------------------------------
# Main App (Tabs)
# ---------------------------------------------------
if st.session_state.df is None:
    st.info("Please upload a data file to begin analysis", icon="ℹ️")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🔍 NL to SQL", "📊 Visualize", "🧐 Data Quality"])

# ----------------------------------
# TAB 1: NATURAL LANGUAGE TO SQL
# ----------------------------------
with tab1:
    st.header("Natural Language → SQL Translator")
    
    if not st.session_state.gemini_api_key:
        st.warning("Please add Gemini API key in sidebar")
        st.stop()

    # User query input
    query = st.text_area(
        "Ask anything about your data",
        placeholder="E.g.: 'Top 5 customers by revenue in California'"
    )

    # Execute analysis
    if st.button("Analyze", type="primary") and query:
        with st.spinner("Processing with Gemini..."):
            try:
                model = genai.GenerativeModel(
                    'gemini-1.5-flash',
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )

                # Step 1: Generate clean SQL query
                sql_prompt = f"""Convert this natural language question to SQL:
                
Question: {query}

Database Schema: {str(st.session_state.df.columns.tolist())}

IMPORTANT:
- Use STANDARD SQL (compatible with SQLite)
- DO NOT include markdown formatting like ```sql ```
- Quote table/column names if they contain spaces
- For counts, use COUNT(*), not COUNT(column)
"""
                sql_response = model.generate_content(sql_prompt)
                clean_sql = sql_response.text.replace("```sql", "").replace("```", "").strip()

                # Step 2: Execute the query
                execution_result = ""
                result_df = None

                if st.session_state.db_conn:
                    # SQLite execution           
                    try:
                        result_df = pd.read_sql(clean_sql, st.session_state.db_conn)
                        execution_result = "✅ Executed on SQLite database"
                    except Exception as e:
                        execution_result = f"❌ SQL Error: {str(e)}"
                else:
                    # CSV/Excel execution using pandasql
                    try:
                        pysqldf = lambda q: sqldf(q, globals())
                        result_df = pysqldf(clean_sql)
                        execution_result = "✅ Executed on flat file data"
                    except Exception as e:
                        execution_result = f"❌ Execution Error: {str(e)}"

                # Step 3: Get explanation
                explain_prompt = f"""Explain this SQL query in simple terms:
                
Query: {clean_sql}

Expected Result: {str(result_df.columns.tolist() if result_df is not None else '')}

Provide:
1. What the query calculates
2. Business meaning of the results
3. Suggested visualization type
"""
                explain_response = model.generate_content(explain_prompt)

                # Store results
                st.session_state.last_analysis = {
                    "query": query,
                    "sql": clean_sql,
                    "execution_result": execution_result,
                    "result_df": result_df,
                    "explanation": explain_response.text
                }

            except Exception as e:
                st.error(f"Gemini Error: {str(e)}")

    # Display results if available
    if 'last_analysis' in st.session_state:
        st.divider()
        
        st.subheader("Generated SQL")
        st.code(st.session_state.last_analysis["sql"], language="sql")
        
        st.subheader("Execution Result")
        st.code(st.session_state.last_analysis["execution_result"])
        
        if st.session_state.last_analysis["result_df"] is not None:
            st.subheader("Query Results")
            st.dataframe(
                st.session_state.last_analysis["result_df"],
                use_container_width=True
            )
        
        st.subheader("Analysis")
        st.markdown(st.session_state.last_analysis["explanation"])

# ----------------------------------
# TAB 2: VISUALIZATION
# ----------------------------------
with tab2:
    st.header("Interactive Visualization")
    
    if 'last_analysis' not in st.session_state or st.session_state.last_analysis["result_df"] is None:
        st.warning("Run a query first to visualize results")
        st.stop()
    
    df = st.session_state.last_analysis["result_df"]
    
    # Visualization controls
    col1, col2 = st.columns(2)
    
    with col1:
        chart_type = st.selectbox(
            "Chart Type",
            ["Bar", "Line", "Scatter", "Histogram", "Pie"],
            index=0
        )
        
    with col2:
        x_axis = st.selectbox(
            "X-Axis",
            df.columns,
            index=0
        )
    
    # Dynamic options
    y_options = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    color_options = [col for col in df.columns if len(df[col].unique()) <= 20]
    
    if len(y_options) == 0:
        st.warning("No numeric columns found for Y-axis")
    else:
        y_axis = st.selectbox(
            "Y-Axis",
            y_options,
            index=0
        )
        
        if len(color_options) > 0:
            color_col = st.selectbox(
                "Color By",
                ["None"] + color_options,
                index=0
            )
        else:
            color_col = "None"
        
        # Generate chart
        try:
            fig = None
            
            if chart_type == "Bar":
                fig = px.bar(
                    df,
                    x=x_axis,
                    y=y_axis,
                    color=None if color_col == "None" else color_col
                )
            elif chart_type == "Line":
                fig = px.line(
                    df,
                    x=x_axis,
                    y=y_axis,
                    color=None if color_col == "None" else color_col
                )
            elif chart_type == "Scatter":
                fig = px.scatter(
                    df,
                    x=x_axis,
                    y=y_axis,
                    color=None if color_col == "None" else color_col
                )
            elif chart_type == "Histogram":
                fig = px.histogram(
                    df,
                    x=x_axis,
                    y=None if x_axis == y_axis else y_axis,
                    color=None if color_col == "None" else color_col
                )
            elif chart_type == "Pie":
                fig = px.pie(
                    df,
                    names=x_axis,
                    values=y_axis
                )
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Chart Error: {str(e)}")

# ----------------------------------
# TAB 3: DATA QUALITY REPORT
# ----------------------------------
with tab3:
    st.header("Data Quality Analysis")
    
    df = st.session_state.df
    
    # Completeness Analysis
    st.subheader("❶ Completeness")
    completeness = (1 - df.isnull().mean()).sort_values()
    st.bar_chart(completeness)
    st.write("Missing values per column:")
    st.dataframe(
        df.isnull().sum().to_frame("Missing Values"),
        use_container_width=True
    )
    
    # Uniqueness Analysis
    st.subheader("❷ Uniqueness")
    uniqueness = (df.nunique() / len(df)).sort_values()
    st.bar_chart(uniqueness)
    st.write("Unique rate per column:")
    st.dataframe(
        uniqueness.to_frame("Unique Rate"),
        use_container_width=True
    )
    
    # Data Types Overview
    st.subheader("❸ Data Types")
    dtype_report = []
    for col in df.columns:
        dtype_report.append({
            "Column": col,
            "Type": str(df[col].dtype),
            "Sample": str(df[col].iloc[0]) if len(df) > 0 else "N/A"
        })
    dtype_df = pd.DataFrame(dtype_report)
    st.dataframe(
        dtype_df,
        use_container_width=True,
        hide_index=True
    )

# ----------------------------------
# FOOTER
# ----------------------------------
st.divider()
st.caption("DataStory Pro | Interview Project | Gemini-Powered Analytics")
