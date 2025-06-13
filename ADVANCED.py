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
import io
import time

# Configure App
st.set_page_config(
    page_title="DataStory Pro",
    page_icon="🤖",
    st.markdown(
    "<span style='font-size: 18px; font-weight: bold;'Created by Vikram Singh</span>",
    unsafe_allow_html=True
)
    layout="wide"
)

st.title("🎯 DataStory Pro")
st.markdown(
    "<span style='font-size: 18px; font-weight: bold;'>Natural Language → SQL Analysis|| Created by Vikram Singh</span>",
    unsafe_allow_html=True
)
st.divider()

# Initialize Session State
if 'df' not in st.session_state:
    st.session_state.df = None
if 'db_conn' not in st.session_state:
    st.session_state.db_conn = None
if 'active_tables' not in st.session_state:
    st.session_state.active_tables = []
if 'multiple_tables_info' not in st.session_state:
    st.session_state.multiple_tables_info = {}
if 'result_df' not in st.session_state:
    st.session_state.result_df = None
if 'file_type' not in st.session_state:
    st.session_state.file_type = None
if 'last_analysis' not in st.session_state:
    st.session_state.last_analysis = None
if 'db_modified' not in st.session_state:
    st.session_state.db_modified = False
if 'temp_db_path' not in st.session_state:
    st.session_state.temp_db_path = None
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

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
    clean_sql = "\n".join([line for line in clean_sql.split("\n") if not line.strip().startswith("--")])
    return clean_sql.strip()


def execute_query(query):
    """Execute SQL query on either SQLite or pandas DataFrame"""
    try:
        if st.session_state.db_conn:  # SQLite database
            query_type = query.strip().split()[0].upper()
            if query_type in ['SELECT', 'WITH']:
                # For SELECT, we read the uncommitted state to preview changes
                return pd.read_sql(query, st.session_state.db_conn)
            else: # DML/DDL Commands
                cursor = st.session_state.db_conn.cursor()
                cursor.execute(query)
                # DO NOT COMMIT HERE. We wait for the user to click "Save".
                st.session_state.db_modified = True # Flag that DB has been changed
                # Refresh table previews from the connection to show pending changes
                refresh_all_tables() 
                if st.session_state.active_tables:
                    st.session_state.df = st.session_state.multiple_tables_info[st.session_state.active_tables[0]]['df']
                st.success("Query executed. Preview updated with pending changes. Click 'Save Changes' to make them permanent.")
                return pd.DataFrame({"Status": [f"Success! Rows affected: {cursor.rowcount}"]})
        else:  # CSV/Excel - use pandasql (only SELECT queries)
            frames = {table_name: table_info['df'] for table_name, table_info in st.session_state.multiple_tables_info.items()}
            frames['df'] = st.session_state.df
            return sqldf(query, frames)
    except Exception as e:
        st.error(f"Failed to execute query: {str(e)}")
        st.error(f"Query: {query}")
        return None

def validate_query(sql_query):
    """Basic validation for SQL queries - ALLOWS DML/DDL COMMANDS"""
    sql_query = sql_query.strip()
    if not sql_query:
        return False, "Query is empty"
    if not re.search(r'\b(?:select|insert|update|delete|create|alter|drop|with|explain)\b', sql_query, re.IGNORECASE):
        return False, "Query doesn't contain valid SQL commands"
    dangerous_keywords = ["file:", "system", "exec", "shell", "http", "ftp", "attach"]
    if any(keyword in sql_query.lower() for keyword in dangerous_keywords):
        return False, "Query contains dangerous operations"
    return True, "Valid query"

def refresh_table(table_name):
    """Refresh a single table's dataframe from the database after modification"""
    if st.session_state.db_conn:
        # Read the full table now for accuracy in visualization
        df_table = pd.read_sql(f"SELECT * FROM {table_name}", st.session_state.db_conn)
        cursor = st.session_state.db_conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        table_info = cursor.fetchall()
        st.session_state.multiple_tables_info[table_name] = {
            'df': df_table,
            'schema': table_info,
            'sample_data': df_table.head(3).to_dict(orient='records')
        }
        st.session_state.last_refresh = time.time()
        return True
    return False

def refresh_all_tables():
    """Refresh all tables after database modification"""
    if st.session_state.db_conn and st.session_state.active_tables:
        for table_name in st.session_state.active_tables:
            refresh_table(table_name)
        return True
    return False

def export_db():
    """Export SQLite database to bytes for download"""
    if st.session_state.db_conn and st.session_state.temp_db_path:
        st.session_state.db_conn.commit()
        buffer = io.BytesIO()
        with open(st.session_state.temp_db_path, 'rb') as f:
            buffer.write(f.read())
        buffer.seek(0)
        return buffer
    return None

def auto_create_chart(df, chart_type="auto", x=None, y=None, color=None):
    """
    Intelligently create a chart based on data types and column selection.
    This is a more robust and simplified version.
    """
    if df.empty:
        st.warning("Cannot create chart from empty data.")
        return None

    # Get column types
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()

    # --- AUTO CHART DETECTION ---
    if chart_type == "auto":
        # Time series analysis
        if len(datetime_cols) > 0 and len(numeric_cols) > 0:
            st.info("💡 Auto-detected time-series data. Plotting a line chart.")
            return px.line(df, x=datetime_cols[0], y=numeric_cols[0], title=f"{numeric_cols[0]} over Time")
        # Categorical vs Numeric
        elif len(categorical_cols) > 0 and len(numeric_cols) > 0:
            st.info("💡 Auto-detected categorical and numeric data. Plotting a bar chart.")
            # Aggregate data for clarity if categories are not unique
            agg_df = df.groupby(categorical_cols[0])[numeric_cols[0]].sum().reset_index()
            return px.bar(agg_df, x=categorical_cols[0], y=numeric_cols[0], title=f"Total {numeric_cols[0]} by {categorical_cols[0]}")
        # Two numeric columns
        elif len(numeric_cols) >= 2:
            st.info("💡 Auto-detected multiple numeric columns. Plotting a scatter chart.")
            return px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title=f"{numeric_cols[0]} vs {numeric_cols[1]}")
        # Single numeric column
        elif len(numeric_cols) == 1:
            st.info("💡 Auto-detected a single numeric column. Plotting a histogram.")
            return px.histogram(df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}")
        # Single categorical column
        elif len(categorical_cols) == 1:
            st.info("💡 Auto-detected a single categorical column. Plotting a pie chart.")
            return px.pie(df, names=categorical_cols[0], title=f"Distribution of {categorical_cols[0]}")
        else:
            st.warning("Auto-detection failed. Please select a chart type and axes manually.")
            return None

    # --- MANUAL CHART SELECTION ---
    try:
        if chart_type == "bar":
            if not x or not y:
                st.warning("Bar charts require X (categorical) and Y (numeric) axes.")
                return None
            return px.bar(df, x=x, y=y, color=color, title=f"{y} by {x}")

        elif chart_type == "line":
            if not x or not y:
                st.warning("Line charts require X (temporal or numeric) and Y (numeric) axes.")
                return None
            return px.line(df, x=x, y=y, color=color, title=f"{y} over {x}")

        elif chart_type == "scatter":
            if not x or not y:
                st.warning("Scatter plots require X and Y numeric axes.")
                return None
            return px.scatter(df, x=x, y=y, color=color, title=f"{y} vs {x}")

        elif chart_type == "pie":
            if not x:
                st.warning("Pie charts require a column for 'Names' (the slices).")
                return None
            return px.pie(df, names=x, values=y, color=color, title=f"Distribution of {x}" + (f" by {y}" if y else ""))

        elif chart_type == "histogram":
            if not x:
                st.warning("Histograms require a numeric column for the X-axis.")
                return None
            return px.histogram(df, x=x, color=color, title=f"Distribution of {x}")
        
        elif chart_type == "box":
            if not y:
                st.warning("Box plots require a numeric column for the Y-axis.")
                return None
            return px.box(df, x=x, y=y, color=color, title=f"Distribution of {y}" + (f" by {x}" if x else ""))

    except Exception as e:
        st.error(f"Failed to create chart: {str(e)}")
        return None

# ----------------------------------
# Sidebar Configuration
# ----------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
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

    model_options = ["gemini-1.5-flash", "gemini-1.5-pro"]
    selected_model = st.selectbox(
        "Model Version",
        model_options,
        index=0,
        help="1.5-flash is fastest, 1.5-pro is more capable"
    )
    
    st.divider()

    st.subheader("📂 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload data file",
        type=["csv", "xlsx", "xls", "db"],
        help="CSV, Excel, or SQLite Database"
    )

    if uploaded_file:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        # Load data only if it's a new file
        if 'uploaded_filename' not in st.session_state or st.session_state.uploaded_filename != uploaded_file.name:
            st.session_state.uploaded_filename = uploaded_file.name
            st.session_state.db_modified = False # Reset modification state on new file upload
            try:
                with st.spinner("Processing file..."):
                    if file_ext == 'csv':
                        st.session_state.df = pd.read_csv(uploaded_file)
                        st.session_state.db_conn = None
                        st.session_state.file_type = file_ext
                        st.success("CSV Loaded Successfully!")
                    elif file_ext in ['xlsx', 'xls']:
                        st.session_state.df = pd.read_excel(uploaded_file)
                        st.session_state.db_conn = None
                        st.session_state.file_type = file_ext
                        st.success("Excel File Loaded Successfully!")
                    elif file_ext == 'db':
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
                            tmp.write(uploaded_file.getvalue())
                            st.session_state.temp_db_path = tmp.name
                        
                        conn = sqlite3.connect(st.session_state.temp_db_path, check_same_thread=False)
                        st.session_state.db_conn = conn
                        st.session_state.file_type = file_ext
                        
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                        tables = [t[0] for t in cursor.fetchall()]
                        st.session_state.all_db_tables = tables
                        
                        # Set default selected tables to all tables
                        if 'active_tables' not in st.session_state or not st.session_state.active_tables:
                             st.session_state.active_tables = tables

            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                st.session_state.df = None

        # UI for DB table selection and preview
        if st.session_state.file_type == 'db':
            st.session_state.active_tables = st.multiselect(
                "Select Tables to Analyze",
                st.session_state.all_db_tables,
                default=st.session_state.active_tables
            )
            # Initial load or refresh of table data into session state
            refresh_all_tables()

            if st.session_state.active_tables:
                preview_table = st.selectbox(
                    "Preview selected table data",
                    st.session_state.active_tables,
                    index=0,
                    key=f"preview_select_{st.session_state.last_refresh}" # Rerender on refresh
                )
                if preview_table in st.session_state.multiple_tables_info:
                    st.dataframe(
                        st.session_state.multiple_tables_info[preview_table]['df'].head(5),
                        use_container_width=True,
                        key=f"preview_df_{preview_table}_{st.session_state.last_refresh}"
                    )
            else:
                 st.warning("Please select at least one table to start.")
                 st.session_state.df = None

        elif st.session_state.df is not None:
            st.write("Preview:")
            st.dataframe(st.session_state.df.head(3), use_container_width=True)

    # --- NEW: Database Management Section ---
    if st.session_state.file_type == 'db' and st.session_state.db_conn:
        st.divider()
        st.subheader("💾 Database Management")

        # Warning for unsaved changes
        if st.session_state.db_modified:
            st.warning("You have unsaved changes!")

        # Save Changes Button
        if st.button("💾 Save Changes", use_container_width=True, type="primary",
                     help="Commit all DML/DDL changes to the database file.",
                     disabled=not st.session_state.db_modified):
            with st.spinner("Saving..."):
                st.session_state.db_conn.commit()
                st.session_state.db_modified = False
                st.success("Changes saved successfully!")
                time.sleep(1) # Give user time to see success message
                st.rerun()

        # Download Modified DB Button
        db_bytes = export_db()
        if db_bytes:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="📥 Download Modified Database",
                data=db_bytes,
                file_name=f"modified_db_{timestamp}.db",
                mime="application/octet-stream",
                use_container_width=True
            )

# ---------------------------------------------------
# Main App (Tabs)
# ---------------------------------------------------
if st.session_state.df is None and not st.session_state.active_tables:
    st.info("Please upload a data file to begin analysis", icon="ℹ️")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🔍 NL to SQL", "📊 Visualize", "🧐 Data Quality"])

# TAB 1: NATURAL LANGUAGE TO SQL
with tab1:
    # ... (No changes needed in this tab, keeping it as is)
    st.header("Natural Language → SQL Translator")
    
    if not st.session_state.gemini_api_key:
        st.warning("Please add Gemini API key in sidebar")
        st.stop()

    query = st.text_area(
        "Ask anything about your data",
        placeholder="E.g.: 'Top 5 customers by revenue in California' OR 'Add new column total_price as price * quantity'",
        height=100
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        run_analysis = st.button("Analyze", type="primary", use_container_width=True)

    if run_analysis and query:
        with st.spinner("Processing..."):
            try:
                model = genai.GenerativeModel(
                    selected_model,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                context = {"file_type": st.session_state.file_type}
                if st.session_state.db_conn:
                    context["database_type"] = "SQLite"
                    context["tables"] = {}
                    for table_name, table_info in st.session_state.multiple_tables_info.items():
                        context["tables"][table_name] = {
                            "columns": table_info['df'].columns.tolist(),
                            "sample_data": table_info['sample_data'],
                            "schema": [(col[1], col[2]) for col in table_info['schema']]
                        }
                else:
                    context["database_type"] = "CSV/Excel"
                    context["table_name"] = "df"
                    context["columns"] = st.session_state.df.columns.tolist()
                    context["sample_data"] = st.session_state.df.head(3).to_dict(orient='records')

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
6. For SELECT queries that are not aggregations, add a LIMIT 1000 clause to avoid excessive output.
7. Use explicit JOIN syntax instead of comma-separated tables
8. Qualify column names with table aliases when joining tables
9. Table names: {list(st.session_state.multiple_tables_info.keys()) if st.session_state.file_type == 'db' else 'df'}
10. For DML/DDL commands (INSERT, UPDATE, DELETE, ALTER, CREATE), ensure they are valid.
"""
                sql_response = model.generate_content(sql_prompt)
                clean_sql = clean_sql_query(sql_response.text)
                is_valid, validation_msg = validate_query(clean_sql)
                if not is_valid:
                    st.error(f"Invalid SQL: {validation_msg}")
                    st.code(clean_sql, language="sql")
                    st.stop()

                result_df = execute_query(clean_sql)
                if result_df is not None:
                    query_type = clean_sql.strip().split()[0].upper()
                    if query_type in ['SELECT', 'WITH']:
                        if result_df.empty:
                            st.warning("Query returned no results.")
                            explanation = "The SQL query executed successfully but returned an empty result set."
                        else:
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
                            explanation = explain_response.text
                    else: # DML/DDL
                        explanation = f"Successfully executed {query_type} command. Database has been modified. Remember to save your changes!"

                    st.session_state.last_analysis = {
                        "query": query,
                        "sql": clean_sql,
                        "result_df": result_df,
                        "explanation": explanation
                    }

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
    
    # Display last analysis
    if 'last_analysis' in st.session_state and st.session_state.last_analysis is not None:
        st.divider()
        st.subheader("Generated SQL")
        st.code(st.session_state.last_analysis["sql"], language="sql")
        
        result_df = st.session_state.last_analysis["result_df"]
        query_type = st.session_state.last_analysis["sql"].strip().split()[0].upper()
        
        if query_type in ['SELECT', 'WITH']:
            st.subheader("Query Results")
            if not result_df.empty:
                st.dataframe(result_df, use_container_width=True, height=300)
            else:
                st.info("The query ran successfully but returned no data.")

        st.subheader("Analysis")
        st.markdown(st.session_state.last_analysis["explanation"])


# --- TAB 2: ROBUST VISUALIZATION ---
with tab2:
    st.header("📊 Interactive Visualization")
    
    # Data source selection
    sources = ["Last Query Result"]
    if st.session_state.file_type == 'db' and st.session_state.active_tables:
        sources.extend([f"DB Table: {name}" for name in st.session_state.active_tables])
    elif st.session_state.file_type != 'db':
        sources.append("Original Data")

    data_source_name = st.selectbox("Select Data to Visualize", sources)
    
    df_to_visualize = None
    if data_source_name == "Last Query Result":
        if st.session_state.last_analysis and not st.session_state.last_analysis["result_df"].empty:
            df_to_visualize = st.session_state.last_analysis["result_df"]
        else:
            st.warning("No query result available to visualize. Please run a query in the 'NL to SQL' tab first.")
    elif data_source_name.startswith("DB Table:"):
        table_name = data_source_name.split(": ")[1]
        if table_name in st.session_state.multiple_tables_info:
            df_to_visualize = st.session_state.multiple_tables_info[table_name]['df']
    elif data_source_name == "Original Data":
        df_to_visualize = st.session_state.df

    if df_to_visualize is None:
        st.stop()

    st.success(f"Visualizing dataset with {len(df_to_visualize)} rows and {len(df_to_visualize.columns)} columns.")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Chart Settings")
        
        chart_type_map = {
            "Auto Detect": "auto", "Bar Chart": "bar", "Line Chart": "line",
            "Scatter Plot": "scatter", "Pie Chart": "pie", "Histogram": "histogram", "Box Plot": "box"
        }
        chart_display_name = st.selectbox("Chart Type", list(chart_type_map.keys()))
        selected_chart_type = chart_type_map[chart_display_name]

        # Get available columns
        all_cols = df_to_visualize.columns.tolist()
        numeric_cols = df_to_visualize.select_dtypes(include='number').columns.tolist()
        
        # Smart axis selection
        x_axis, y_axis, color = None, None, None
        
        if selected_chart_type in ["bar", "line", "scatter", "box"]:
            x_axis = st.selectbox("X-Axis", [""] + all_cols)
            y_axis = st.selectbox("Y-Axis", [""] + numeric_cols)
        elif selected_chart_type == "pie":
            x_axis = st.selectbox("Slice Names (Names)", [""] + all_cols)
            y_axis = st.selectbox("Slice Values (Optional)", [""] + numeric_cols)
        elif selected_chart_type == "histogram":
            x_axis = st.selectbox("Numeric Column", [""] + numeric_cols)
        
        if selected_chart_type != "auto":
            color = st.selectbox("Color By (Optional)", [""] + all_cols)

    with col2:
        st.subheader("Chart Preview")
        
        # Ensure selections are not empty strings
        x_param = x_axis if x_axis else None
        y_param = y_axis if y_axis else None
        color_param = color if color else None

        fig = auto_create_chart(
            df_to_visualize,
            chart_type=selected_chart_type,
            x=x_param,
            y=y_param,
            color=color_param
        )
        
        if fig:
            fig.update_layout(height=600, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select a chart type and axes to generate a visualization.")

# TAB 3: DATA QUALITY REPORT
with tab3:
    # ... (No changes needed in this tab, keeping it as is)
    st.header("Data Quality Analysis")
    
    df_for_quality = None
    if st.session_state.file_type == 'db' and st.session_state.active_tables:
        selected_table = st.selectbox("Select table for quality analysis", st.session_state.active_tables, key="quality_table_select")
        if selected_table in st.session_state.multiple_tables_info:
            df_for_quality = st.session_state.multiple_tables_info[selected_table]['df']
    else:
        df_for_quality = st.session_state.df

    if df_for_quality is None:
        st.warning("No data selected for quality analysis.")
        st.stop()
        
    st.subheader("❶ Completeness Check")
    completeness = (1 - df_for_quality.isnull().mean()).sort_values()
    st.bar_chart(completeness, color="#2EC771")
    
    with st.expander("Show Detailed Missing Values"):
        missing_data = df_for_quality.isnull().sum().to_frame("Missing Values")
        missing_data["% Missing"] = (missing_data["Missing Values"] / len(df_for_quality) * 100).round(2)
        st.dataframe(missing_data[missing_data["Missing Values"] > 0], use_container_width=True)
    
    st.subheader("❷ Patterns & Distributions")
    col1, col2 = st.columns(2)
    with col1:
        num_cols = df_for_quality.select_dtypes(include='number').columns.tolist()
        if num_cols:
            selected_num = st.selectbox("Numeric Column", num_cols)
            fig_num = go.Figure()
            fig_num.add_trace(go.Histogram(x=df_for_quality[selected_num], name="Histogram"))
            fig_num.add_trace(go.Box(x=df_for_quality[selected_num], name="Box Plot", marker_color='red'))
            fig_num.update_layout(title=f"Distribution of {selected_num}", height=400, template="plotly_white")
            st.plotly_chart(fig_num, use_container_width=True)
    
    with col2:
        cat_cols = df_for_quality.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            selected_cat = st.selectbox("Categorical Column", cat_cols)
            value_counts = df_for_quality[selected_cat].value_counts().head(20)
            fig_cat = px.bar(
                value_counts,
                y=value_counts.index,
                x=value_counts.values,
                orientation='h',
                labels={'y': selected_cat, 'x': 'Count'},
                title=f"Top {len(value_counts)} Values for {selected_cat}"
            )
            fig_cat.update_layout(height=400, template="plotly_white", yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_cat, use_container_width=True)
    
    st.subheader("❸ Schema Information")
    st.write(f"Total Rows: {len(df_for_quality):,} | Columns: {len(df_for_quality.columns)}")
    schema_list = []
    for col in df_for_quality.columns:
        dtype = str(df_for_quality[col].dtype)
        unique_vals = df_for_quality[col].nunique()
        sample_val = ""
        if not df_for_quality[col].dropna().empty:
            sample_val = str(df_for_quality[col].dropna().iloc[0])
        
        schema_list.append({
            "Column": col, "Type": dtype, 
            "Unique Values": f"{unique_vals:,}", 
            "Sample": sample_val[:75] + '...' if len(sample_val) > 75 else sample_val
        })
    st.dataframe(pd.DataFrame(schema_list), use_container_width=True, hide_index=True)
    
    st.subheader("❹ Correlation Analysis")
    num_cols_corr = df_for_quality.select_dtypes(include='number').columns.tolist()
    if len(num_cols_corr) > 1:
        corr_method = st.selectbox("Correlation Method", ["pearson", "spearman", "kendall"])
        corr_matrix = df_for_quality[num_cols_corr].corr(method=corr_method)
        fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", title="Correlation Matrix", color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Not enough numeric columns for correlation analysis (requires at least 2).")

# ----------------------------------
# FOOTER
# ----------------------------------
st.divider()
st.caption("DataStory Pro | Created by Vikram Singh")
