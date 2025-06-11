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
if 'file_type' not in st.session_state:
    st.session_state.file_type = None

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
# Helper Functions
# ----------------------------------
def clean_sql_query(sql_text):
    """Remove markdown formatting and clean SQL queries"""
    clean_sql = sql_text.replace("```sql", "").replace("```", "").strip()
    # Remove SQL comments
    clean_sql = "\n".join([line for line in clean_sql.split("\n") if not line.strip().startswith("--")])
    return clean_sql.strip()

def execute_query(query, df_name="df"):
    """Execute SQL query on either SQLite or pandas DataFrame"""
    try:
        if st.session_state.db_conn:  # SQLite database
            return pd.read_sql(query, st.session_state.db_conn)
        else:  # CSV/Excel - use pandasql
            pysqldf = lambda q: sqldf(q, globals())
            return pysqldf(query)
    except Exception as e:
        st.error(f"Failed to execute query: {str(e)}")
        return None

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

    # File Upload Section
    st.subheader("📂 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload data file",
        type=["csv", "xlsx", "xls", "db"],
        help="CSV, Excel, or SQLite Database"
    )

    if uploaded_file:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        st.session_state.file_type = file_ext
        
        try:
            with st.spinner("Processing file..."):
                if file_ext == 'csv':
                    st.session_state.df = pd.read_csv(uploaded_file)
                    st.session_state.db_conn = None
                    st.success("CSV Loaded Successfully!")

                elif file_ext in ['xlsx', 'xls']:
                    st.session_state.df = pd.read_excel(uploaded_file)
                    st.session_state.db_conn = None
                    st.success("Excel File Loaded Successfully!")

                elif file_ext == 'db':
                    # Save DB to temp file
                    with tempfile.NamedTemporaryFile(delete=False) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    # Connect to SQLite
                    conn = sqlite3.connect(tmp_path)
                    st.session_state.db_conn = conn

                    # Show available tables
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    
                    if tables:
                        table_names = [t[0] for t in tables]
                        selected_table = st.selectbox(
                            "Select Table to Analyze",
                            table_names,
                            index=0
                        )
                        st.session_state.active_table = selected_table
                        st.session_state.df = pd.read_sql(
                            f"SELECT * FROM {selected_table} LIMIT 1000",
                            conn
                        )

                        # Get table schema
                        cursor.execute(f"PRAGMA table_info({selected_table})")
                        st.session_state.table_info = cursor.fetchall()
                        st.success(f"Loaded Table: {selected_table}")

                    os.unlink(tmp_path)

            # Show sample data
            if st.session_state.df is not None:
                st.write("Preview:")
                st.dataframe(st.session_state.df.head(3), use_container_width=True)

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

    st.divider()
    st.caption("💡 For SQLite files, select the table to analyze")

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
        placeholder="E.g.: 'Top 5 customers by revenue in California'",
        height=100
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        run_analysis = st.button("Analyze", type="primary", use_container_width=True)

    # Execute analysis
    if run_analysis and query:
        with st.spinner("Processing..."):
            try:
                model = genai.GenerativeModel(
                    'gemini-1.5-flash',
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )

                # Prepare context for Gemini
                context = {
                    "columns": st.session_state.df.columns.tolist(),
                    "sample_data": st.session_state.df.head(3).to_dict(orient='records'),
                    "file_type": st.session_state.file_type
                }

                if st.session_state.db_conn:
                    context["database_type"] = "SQLite"
                    context["table_name"] = st.session_state.active_table
                    context["schema"] = [(col[1], col[2]) for col in st.session_state.table_info]

                # Step 1: Generate clean SQL query
                sql_prompt = f"""Convert this natural language question to valid SQL:
                
QUESTION: {query}

DATA CONTEXT:
{context}

GUIDELINES:
1. Use STANDARD SQL (compatible with SQLite)
2. DO NOT include markdown formatting
3. If working with CSV/Excel, use 'df' as the table name
4. Quote identifiers only when needed (spaces/special chars)
5. For counts, prefer COUNT(*) unless specific column needed
6. Always include a LIMIT clause unless explicitly asked for totals
"""
                sql_response = model.generate_content(sql_prompt)
                clean_sql = clean_sql_query(sql_response.text)
                
                # Modify query if CSV/Excel is used
                if not st.session_state.db_conn and "FROM" in clean_sql:
                    clean_sql = clean_sql.replace("FROM \"", "FROM ").replace("FROM ", "FROM df ")

                # Step 2: Execute the query
                result_df = execute_query(clean_sql)

                # Step 3: Get explanation
                if result_df is not None:
                    explain_prompt = f"""Explain this SQL query in business terms:
                    
QUERY: {clean_sql}

RESULTS COLUMNS: {result_df.columns.tolist()}

Provide:
1. Plain English explanation of what calculates
2. Business meaning of the results
3. Suggested visualization type
4. Potential limitations or caveats
"""
                    explain_response = model.generate_content(explain_prompt)

                    # Store results
                    st.session_state.last_analysis = {
                        "query": query,
                        "sql": clean_sql,
                        "result_df": result_df,
                        "explanation": explain_response.text
                    }

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

    # Display results if available
    if 'last_analysis' in st.session_state:
        st.divider()
        
        st.subheader("Generated SQL")
        st.code(st.session_state.last_analysis["sql"], language="sql")
        
        st.subheader("Query Results")
        st.dataframe(
            st.session_state.last_analysis["result_df"],
            use_container_width=True,
            height=300
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        chart_type = st.selectbox(
            "Chart Type",
            ["Bar", "Line", "Scatter", "Histogram", "Pie", "Box"],
            index=0
        )
        
    with col2:
        x_axis = st.selectbox(
            "X-Axis",
            df.columns,
            index=0
        )
    
    # Dynamic options
    y_options = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])] or ["(None)"]
    y_axis = st.selectbox("Y-Axis", y_options, index=0 if len(y_options) > 0 else 0)
    
    color_options = ["None"] + [col for col in df.columns if len(df[col].unique()) <= 10]
    color_col = st.selectbox("Color By", color_options, index=0)
    
    # Generate visualizations
    if chart_type == "Bar":
        fig = px.bar(
            df,
            x=x_axis,
            y=None if y_axis == "(None)" else y_axis,
            color=None if color_col == "None" else color_col
        )
    elif chart_type == "Line":
        fig = px.line(
            df,
            x=x_axis,
            y=None if y_axis == "(None)" else y_axis,
            color=None if color_col == "None" else color_col
        )
    elif chart_type == "Scatter":
        fig = px.scatter(
            df,
            x=x_axis,
            y=None if y_axis == "(None)" else y_axis,
            color=None if color_col == "None" else color_col
        )
    elif chart_type == "Histogram":
        fig = px.histogram(
            df,
            x=x_axis,
            y=None if y_axis == "(None)" else y_axis,
            color=None if color_col == "None" else color_col
        )
    elif chart_type == "Pie":
        fig = px.pie(
            df,
            names=x_axis,
            values=y_axis if y_axis != "(None)" else df[x_axis].value_counts(),
            color=None if color_col == "None" else color_col
        )
    elif chart_type == "Box":
        fig = px.box(
            df,
            x=x_axis,
            y=None if y_axis == "(None)" else y_axis,
            color=None if color_col == "None" else color_col
        )
    
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------
# TAB 3: DATA QUALITY REPORT
# ----------------------------------
with tab3:
    st.header("Data Quality Analysis")
    
    df = st.session_state.df
    
    # Completeness Analysis
    st.subheader("❶ Completeness Check")
    st.write("Percentage of non-null values per column:")
    completeness = (1 - df.isnull().mean()).sort_values()
    st.bar_chart(completeness)
    
    with st.expander("Show Detailed Missing Values"):
        st.dataframe(
            df.isnull().sum().to_frame("Missing Values"),
            use_container_width=True
        )
    
    # Patterns & Distributions
    st.subheader("❷ Patterns & Distributions")
    
    col1, col2 = st.columns(2)
    with col1:
        num_cols = df.select_dtypes(include='number').columns.tolist()
        if num_cols:
            selected_num = st.selectbox("Numeric Column", num_cols)
            st.line_chart(df[selected_num])
    
    with col2:
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        if cat_cols:
            selected_cat = st.selectbox("Categorical Column", cat_cols)
            st.bar_chart(df[selected_cat].value_counts().head(10))
    
    # Schema Information
    st.subheader("❸ Schema Information")
    st.write(f"Total Rows: {len(df)} | Columns: {len(df.columns)}")
    
    schema = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        unique = df[col].nunique()
        sample = str(df[col].iloc[0]) if len(df) > 0 else "N/A"
        
        schema.append({
            "Column": col,
            "Type": dtype,
            "Unique Values": unique,
            "Sample": sample[:50] + "..." if len(sample) > 50 else sample
        })
    
    st.dataframe(
        pd.DataFrame(schema),
        use_container_width=True,
        hide_index=True
    )

# ----------------------------------
# FOOTER
# ----------------------------------
st.divider()
st.caption("DataStory Pro | Code: github.com/yourusername | Powered by Google Gemini")
