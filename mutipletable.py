import streamlit as st
import pandas as pd
import google.generativeai as genai
import sqlite3
import tempfile
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pandasql import sqldf
import re

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
if 'active_tables' not in st.session_state:  # Changed to plural
    st.session_state.active_tables = []
if 'multiple_tables_info' not in st.session_state:  # Store multiple tables info
    st.session_state.multiple_tables_info = {}
if 'result_df' not in st.session_state:
    st.session_state.result_df = None
if 'file_type' not in st.session_state:
    st.session_state.file_type = None
if 'last_analysis' not in st.session_state:
    st.session_state.last_analysis = None

# Gemini Configuration
generation_config = {
    "temperature": 0.2,
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

def execute_query(query):
    """Execute SQL query on either SQLite or pandas DataFrame"""
    try:
        if st.session_state.db_conn:  # SQLite database
            return pd.read_sql(query, st.session_state.db_conn)
        else:  # CSV/Excel - use pandasql
            # Use the union of all DataFrames
            frames = {table_name: table_info['df'] for table_name, table_info in st.session_state.multiple_tables_info.items()}
            frames['df'] = st.session_state.df  # For CSV/Excel single table
            return sqldf(query, frames)
    except Exception as e:
        st.error(f"Failed to execute query: {str(e)}")
        st.error(f"Query: {query}")
        return None

def validate_query(sql_query):
    """Basic validation for SQL queries"""
    sql_query = sql_query.lower().strip()
    
    # Block potentially dangerous operations
    dangerous_keywords = ["drop", "delete", "insert", "update", "alter", "create", "replace"]
    if any(keyword in sql_query for keyword in dangerous_keywords):
        return False, "Query contains dangerous operations"
    
    # Ensure SELECT is present
    if not sql_query.startswith("select"):
        return False, "Only SELECT queries are allowed"
    
    return True, "Valid query"

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

    # Model selection
    model_options = ["gemini-1.5-flash", "gemini-1.5-pro"]
    selected_model = st.selectbox(
        "Model Version",
        model_options,
        index=0,
        help="1.5-flash is fastest, 1.5-pro is more capable"
    )
    
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
                        
                        # MULTI-SELECT CHANGE: Replaced selectbox with multiselect
                        selected_tables = st.multiselect(
                            "Select Tables to Analyze",
                            table_names,
                            default=table_names[0] if table_names else None
                        )
                        
                        st.session_state.active_tables = selected_tables
                        st.session_state.multiple_tables_info = {}
                        
                        # Clear previous single table reference
                        if 'active_table' in st.session_state:
                            del st.session_state.active_table
                        
                        for table_name in selected_tables:
                            # Load each table as a separate DataFrame
                            df_table = pd.read_sql(
                                f"SELECT * FROM {table_name} LIMIT 1000",
                                conn
                            )
                            
                            # Get table schema
                            cursor.execute(f"PRAGMA table_info({table_name})")
                            table_info = cursor.fetchall()
                            
                            # Store information about the table
                            st.session_state.multiple_tables_info[table_name] = {
                                'df': df_table,
                                'schema': table_info,
                                'sample_data': df_table.head(3).to_dict(orient='records')
                            }
                        
                        # Set the default display DataFrame to the first selected table
                        if selected_tables:
                            st.session_state.df = st.session_state.multiple_tables_info[selected_tables[0]]['df']
                            st.success(f"Loaded {len(selected_tables)} tables")
                        else:
                            st.warning("No tables selected")
                            st.session_state.df = None

                    os.unlink(tmp_path)

            # Show sample data
            if st.session_state.df is not None and st.session_state.file_type == 'db':
                # TABLE SELECTOR: Let user choose which table to preview
                preview_table = st.selectbox(
                    "Preview selected table",
                    st.session_state.active_tables,
                    index=0
                )
                st.dataframe(
                    st.session_state.multiple_tables_info[preview_table]['df'].head(3),
                    use_container_width=True
                )
            elif st.session_state.df is not None:
                st.write("Preview:")
                st.dataframe(st.session_state.df.head(3), use_container_width=True)

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

    st.divider()
    st.caption("💡 For SQLite files, select one or more tables to analyze")

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
                    selected_model,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )

                # Prepare context for Gemini - handle multiple tables
                context = {
                    "file_type": st.session_state.file_type
                }

                if st.session_state.db_conn:  # SQLite database
                    context["database_type"] = "SQLite"
                    context["tables"] = {}
                    
                    # MULTI-TABLE: Add info for all selected tables
                    for table_name, table_info in st.session_state.multiple_tables_info.items():
                        context["tables"][table_name] = {
                            "columns": table_info['df'].columns.tolist(),
                            "sample_data": table_info['sample_data'],
                            "schema": [(col[1], col[2]) for col in table_info['schema']]
                        }
                else:  # CSV/Excel
                    context["database_type"] = "CSV/Excel"
                    context["table_name"] = "df"
                    context["columns"] = st.session_state.df.columns.tolist()
                    context["sample_data"] = st.session_state.df.head(3).to_dict(orient='records')

                # Step 1: Generate clean SQL query
                sql_prompt = f"""Convert this natural language question to valid SQL:
                
QUESTION: {query}

DATA CONTEXT:
{context}

GUIDELINES:
1. Use STANDARD SQL (compatible with SQLite)
2. DO NOT include markdown formatting
3. Include required JOIN statements when using multiple tables
4. Quote identifiers only when needed (spaces/special chars)
5. For counts, prefer COUNT(*) unless specific column needed
6. Always include a LIMIT clause unless explicitly asked for totals
7. Use explicit JOIN syntax instead of comma-separated tables
8. Qualify column names with table aliases when joining tables
9. Table names: {list(st.session_state.multiple_tables_info.keys()) if st.session_state.file_type == 'db' else 'df'}
"""
                sql_response = model.generate_content(sql_prompt)
                clean_sql = clean_sql_query(sql_response.text)
                
                # Validate SQL before execution
                is_valid, validation_msg = validate_query(clean_sql)
                if not is_valid:
                    st.error(f"Invalid SQL: {validation_msg}")
                    st.code(clean_sql, language="sql")
                    st.stop()

                # Step 2: Execute the query
                result_df = execute_query(clean_sql)

                # Step 3: Get explanation
                if result_df is not None:
                    if result_df.empty:
                        st.warning("Query returned no results")
                        st.stop()

                    explain_prompt = f"""Explain this SQL query in business terms:
                    
QUERY: {clean_sql}

RESULTS COLUMNS: {result_df.columns.tolist()}
SAMPLE RESULTS: {result_df.head(3).to_dict(orient='records')}

Provide:
1. Plain English explanation of what the query calculates
2. Business meaning of the results
3. Suggested visualization types (2-3 options)
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
    if 'last_analysis' in st.session_state and st.session_state.last_analysis is not None:
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
    
    # Data source selection
    sources = ["Original Data", "Last Query Result"]
    if st.session_state.file_type == 'db':
        sources += [f"Table: {name}" for name in st.session_state.active_tables]
    
    data_source = st.radio("Data Source", sources, horizontal=True)
    
    if data_source == "Original Data":
        df = st.session_state.df
    elif st.session_state.last_analysis and st.session_state.last_analysis["result_df"] is not None and data_source == "Last Query Result":
        df = st.session_state.last_analysis["result_df"]
    elif data_source.startswith("Table:") and st.session_state.file_type == 'db':
        table_name = data_source.split(": ")[1]
        df = st.session_state.multiple_tables_info[table_name]['df']
    else:
        st.warning("Run a query first to visualize results")
        st.stop()
    
    # Chart configuration
    col1, col2 = st.columns(2)
    
    with col1:
        chart_type = st.selectbox(
            "Chart Type",
            ["Bar", "Line", "Area", "Scatter", "Pie", "Histogram", "Box", "Violin", "Heatmap", "Sunburst"],
            index=0
        )
        
    with col2:
        x_axis = st.selectbox(
            "X-Axis",
            df.columns,
            index=0
        )
    
    # Dynamic options based on chart type
    if chart_type in ["Bar", "Line", "Scatter", "Area", "Box", "Violin"]:
        y_options = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])] or ["(None)"]
        y_axis = st.selectbox("Y-Axis", y_options, index=0 if len(y_options) > 0 else 0)
    
    if chart_type in ["Scatter", "Bubble"]:
        size_options = ["(None)"] + [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        size_col = st.selectbox("Size", size_options, index=0)
    
    color_options = ["(None)"] + df.columns.tolist()
    color_col = st.selectbox("Color", color_options, index=0)
    
    if chart_type in ["Histogram", "Box", "Violin"]:
        group_options = ["(None)"] + [col for col in df.columns if df[col].nunique() < 20]
        group_col = st.selectbox("Group By", group_options, index=0)
    
    # Advanced options
    with st.expander("Advanced Options"):
        agg_func = st.selectbox("Aggregation", 
                               ["count", "sum", "avg", "min", "max"], 
                               index=1) if chart_type != "Scatter" else "raw"
        
        if chart_type == "Bar":
            orientation = st.radio("Orientation", ["Vertical", "Horizontal"], horizontal=True)
            barmode = st.radio("Bar Mode", ["group", "stack", "relative"], horizontal=True)
        
        if chart_type == "Heatmap":
            z_options = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            z_axis = st.selectbox("Z-Axis (Values)", z_options)
            agg_func_heatmap = st.selectbox("Aggregation Function", 
                                          ["sum", "avg", "count", "min", "max"],
                                          index=1)
    
    # Generate visualizations
    try:
        if chart_type == "Bar":
            fig = px.bar(
                df,
                x=x_axis,
                y=y_axis if y_axis != "(None)" else None,
                color=color_col if color_col != "(None)" else None,
                orientation="h" if orientation == "Horizontal" else "v",
                barmode=barmode
            )
            
        elif chart_type == "Line":
            fig = px.line(
                df,
                x=x_axis,
                y=y_axis if y_axis != "(None)" else None,
                color=color_col if color_col != "(None)" else None
            )
            
        elif chart_type == "Area":
            fig = px.area(
                df,
                x=x_axis,
                y=y_axis if y_axis != "(None)" else None,
                color=color_col if color_col != "(None)" else None
            )
            
        elif chart_type == "Scatter":
            fig = px.scatter(
                df,
                x=x_axis,
                y=y_axis if y_axis != "(None)" else None,
                color=color_col if color_col != "(None)" else None,
                size=size_col if size_col != "(None)" else None,
                hover_name=df.index,
                hover_data=df.columns
            )
            
        elif chart_type == "Pie":
            fig = px.pie(
                df,
                names=x_axis,
                values=y_axis if y_axis != "(None)" else df[x_axis].value_counts(),
                color=color_col if color_col != "(None)" else None
            )
            
        elif chart_type == "Histogram":
            fig = px.histogram(
                df,
                x=x_axis,
                y=y_axis if y_axis != "(None)" else None,
                color=group_col if group_col != "(None)" else None,
                nbins=50,
                marginal="rug"
            )
            
        elif chart_type == "Box":
            fig = px.box(
                df,
                x=group_col if group_col != "(None)" else None,
                y=x_axis,
                color=color_col if color_col != "(None)" else None
            )
            
        elif chart_type == "Violin":
            fig = px.violin(
                df,
                x=group_col if group_col != "(None)" else None,
                y=x_axis,
                color=color_col if color_col != "(None)" else None,
                box=True
            )
            
        elif chart_type == "Heatmap":
            pivot_df = df.pivot_table(
                index=x_axis, 
                columns=color_col, 
                values=z_axis, 
                aggfunc=agg_func_heatmap
            ).fillna(0)
            
            fig = px.imshow(
                pivot_df,
                labels=dict(x=color_col, y=x_axis, color=agg_func_heatmap),
                x=pivot_df.columns.tolist(),
                y=pivot_df.index.tolist()
            )
            
        elif chart_type == "Sunburst":
            path = st.multiselect("Hierarchy Path", df.columns, default=df.columns[:2])
            if len(path) < 2:
                st.warning("Select at least 2 columns for hierarchy")
            else:
                fig = px.sunburst(
                    df,
                    path=path,
                    values=y_axis if y_axis != "(None)" else None,
                    color=color_col if color_col != "(None)" else None
                )
                
        # Update layout
        fig.update_layout(
            height=600,
            title=f"{chart_type} Chart of {x_axis}",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Failed to create chart: {str(e)}")

# ----------------------------------
# TAB 3: DATA QUALITY REPORT
# ----------------------------------
with tab3:
    st.header("Data Quality Analysis")
    
    # TABLE SELECTOR: For SQLite databases
    if st.session_state.file_type == 'db' and st.session_state.active_tables:
        selected_table = st.selectbox(
            "Select table for quality analysis", 
            st.session_state.active_tables,
            index=0
        )
        if selected_table in st.session_state.multiple_tables_info:
            df = st.session_state.multiple_tables_info[selected_table]['df']
        else:
            df = st.session_state.df
            st.warning("Using first table selected")
    else:
        df = st.session_state.df
    
    # Completeness Analysis
    st.subheader("❶ Completeness Check")
    completeness = (1 - df.isnull().mean()).sort_values()
    st.bar_chart(completeness)
    
    with st.expander("Show Detailed Missing Values"):
        missing_data = df.isnull().sum().to_frame("Missing Values")
        missing_data["% Missing"] = (missing_data["Missing Values"] / len(df) * 100).round(2)
        st.dataframe(missing_data, use_container_width=True)
    
    # Patterns & Distributions
    st.subheader("❷ Patterns & Distributions")
    
    col1, col2 = st.columns(2)
    with col1:
        num_cols = df.select_dtypes(include='number').columns.tolist()
        if num_cols:
            selected_num = st.selectbox("Numeric Column", num_cols)
            fig_num = go.Figure()
            fig_num.add_trace(go.Histogram(x=df[selected_num], name="Histogram"))
            fig_num.add_trace(go.Box(x=df[selected_num], name="Box Plot"))
            fig_num.update_layout(title=f"Distribution of {selected_num}", height=400)
            st.plotly_chart(fig_num, use_container_width=True)
    
    with col2:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            selected_cat = st.selectbox("Categorical Column", cat_cols)
            value_counts = df[selected_cat].value_counts().head(20)
            fig_cat = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                labels={'x': selected_cat, 'y': 'Count'},
                title=f"Top {len(value_counts)} {selected_cat} Values"
            )
            fig_cat.update_layout(height=400)
            st.plotly_chart(fig_cat, use_container_width=True)
    
    # Schema Information
    st.subheader("❸ Schema Information")
    st.write(f"Total Rows: {len(df):,} | Columns: {len(df.columns)}")
    
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
    
    # Correlation Analysis
    st.subheader("❹ Correlation Analysis")
    num_cols = df.select_dtypes(include='number').columns.tolist()
    if len(num_cols) > 1:
        corr_method = st.selectbox("Correlation Method", ["pearson", "spearman", "kendall"])
        corr_matrix = df[num_cols].corr(method=corr_method)
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Not enough numeric columns for correlation analysis")

# ----------------------------------
# FOOTER
# ----------------------------------
st.divider()
st.caption("DataStory Pro | Powered by Google Gemini")
