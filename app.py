import streamlit as st
import google.generativeai as genai
import sqlite3
import pandas as pd
import plotly.express as px
import os
from contextlib import closing


# --- Configuration ---
st.set_page_config(
    page_title="NLQ to SQL",
    page_icon="🤖",
    layout="wide"
)

# Configure the Gemini API key from Streamlit's secrets
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

if not GOOGLE_API_KEY:
    st.error("Set GOOGLE_API_KEY in .env and restart. See README instructions.")
    st.stop()


# --- Model and Prompt Configuration ---
MODEL_NAME = "gemini-1.5-flash-latest"
PROMPT_TEMPLATE = """
You are an expert in converting English questions to SQLite queries.
You will be given a database schema and a user's question. Your task is to generate a valid SQLite query that answers the question.

**Database Schema:**
{schema}

**Instructions:**
1.  Analyze the schema to understand the table structures, columns, and relationships.
2.  Generate a SQLite query that is syntactically correct and retrieves the information requested by the user.
3.  **DO NOT** include any explanations, comments, or markdown formatting (like ```sql).
4.  The query should directly answer the question. Do not return any text other than the SQL query itself.
5.  If the question cannot be answered with the given schema, return the single word: "INVALID".
6. User may have asked in very short, or improper information, build a sql query that fits the context.

**User Question:**
"{question}"

**Generated SQLite Query:**
"""

# --- Helper Functions ---

def get_db_connection(db_file):
    """
    Establishes a connection to the SQLite database.
    Saves the uploaded file to a temporary location to get a stable file path.
    """
    if db_file is not None:
        # Save the uploaded file to a temporary file
        temp_dir = "temp_db"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        file_path = os.path.join(temp_dir, db_file.name)
        
        with open(file_path, "wb") as f:
            f.write(db_file.getbuffer())
            
        try:
            conn = sqlite3.connect(file_path)
            return conn
        except sqlite3.Error as e:
            st.error(f"Database connection error: {e}")
            return None
    return None

def get_db_schema(conn):
    """
    Extracts the database schema (table names, columns, and types).
    """
    schema_str = ""
    try:
        with closing(conn.cursor()) as cursor:
            # Get list of tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            for table_name in tables:
                table_name = table_name[0]
                schema_str += f"Table `{table_name}`:\n"
                
                # Get column information for each table
                cursor.execute(f"PRAGMA table_info(`{table_name}`);")
                columns = cursor.fetchall()
                for column in columns:
                    schema_str += f"- `{column[1]}` ({column[2]})\n"
                schema_str += "\n"
    except sqlite3.Error as e:
        st.error(f"Schema extraction error: {e}")
        return None
        
    return schema_str.strip()

def get_gemini_response(schema, question):
    """
    Generates the SQL query using the Gemini Pro model.
    """
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        prompt = PROMPT_TEMPLATE.format(schema=schema, question=question)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"🤖 Gemini API Error: {e}")
        st.info("Please check your API key and ensure the Gemini API is enabled.")
        return None

def execute_sql_query(query, conn):
    """
    Executes the given SQL query and returns the result as a pandas DataFrame.
    """
    # Check for known invalid outputs from the model
    invalid_keywords = ["schema", "invalid"]
    if query and query.strip().lower() in invalid_keywords:
        return None, "The model returned an invalid keyword. Please try rephrasing your question."
    
    try:
        df = pd.read_sql_query(query, conn)
        return df, None
    except (pd.io.sql.DatabaseError, sqlite3.Error) as e:
        return None, f"SQL Execution Error: {e}"

def get_chart_type(df):
    """

    Determines the most suitable chart type based on the dataframe columns.
    Simple logic: Prefers bar for categorical/numeric, line for time-series like data.
    """
    if df.empty or len(df.columns) < 2:
        return None

    first_col_type = df.iloc[:, 0].dtype
    second_col_type = df.iloc[:, 1].dtype

    if pd.api.types.is_numeric_dtype(second_col_type):
        if pd.api.types.is_categorical_dtype(first_col_type) or pd.api.types.is_object_dtype(first_col_type):
            return "bar"
        if pd.api.types.is_datetime64_any_dtype(first_col_type) or pd.api.types.is_numeric_dtype(first_col_type):
             return "line"
        if len(df) < 15: # Pie chart for small number of categories
            return "pie"
            
    return None

def plot_data(df, chart_type):
    """
    Generates a plot using Plotly based on the determined chart type.
    """
    if df.empty or chart_type is None or len(df.columns) < 2:
        return

    x_col, y_col = df.columns[0], df.columns[1]
    
    st.write(f"### 📊 Data Visualization ({chart_type.capitalize()} Chart)")

    try:
        if chart_type == "bar":
            fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}", template="seaborn")
        elif chart_type == "line":
            fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}", template="seaborn")
        elif chart_type == "pie":
            fig = px.pie(df, names=x_col, values=y_col, title=f"Distribution of {y_col} by {x_col}")
        else:
            return

        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not generate chart. Error: {e}")

# --- Streamlit App UI ---

st.title(" Natural Language Query to SQL with Gemini Pro 🚀")
st.write("Upload your SQLite database, ask a question in plain English, and get the answer!")

# --- Sidebar for DB Upload and Schema Display ---
with st.sidebar:
    st.header("Database Setup")
    uploaded_file = st.file_uploader("Upload a SQLite Database", type=["db", "sqlite", "sqlite3"])

    if uploaded_file:
        conn = get_db_connection(uploaded_file)
        if conn:
            st.session_state.db_conn = conn
            st.success("Database connected successfully! ✅")
            
            schema = get_db_schema(st.session_state.db_conn)
            if schema:
                st.session_state.db_schema = schema
                with st.expander("Database Schema", expanded=False):
                    st.text_area("Schema:", value=schema, height=300, disabled=True)
            else:
                st.error("Failed to retrieve database schema.")
                st.stop()
        else:
            st.error("Failed to connect to the database.")
            st.stop()
    else:
        st.info("Please upload a database file to begin.")

# --- Main Area for Query and Results ---
if "db_schema" in st.session_state:
    st.header("Ask your question 💬")
    user_question = st.text_input("Enter your question about the data:", placeholder="e.g., Show me the top 5 artists by total sales")

    if st.button("Generate & Execute Query"):
        if not user_question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("🤖 Gemini is thinking..."):
                generated_sql = get_gemini_response(st.session_state.db_schema, user_question)

                if generated_sql:
                    st.session_state.generated_sql = generated_sql
                    st.subheader("🔍 Generated SQL Query")
                    st.code(generated_sql, language="sql")

                    # Execute the query
                    df, error = execute_sql_query(generated_sql, st.session_state.db_conn)
                    st.session_state.query_result = df
                    st.session_state.query_error = error
                else:
                    st.session_state.generated_sql = None
                    st.session_state.query_result = None
                    st.session_state.query_error = "Failed to generate SQL query."

    # Display results or errors from the last execution
    if "generated_sql" in st.session_state and st.session_state.generated_sql:
        if st.session_state.query_error:
            st.error(st.session_state.query_error)
        elif st.session_state.query_result is not None:
            if st.session_state.query_result.empty:
                st.info("The query executed successfully but returned no results.")
            else:
                st.subheader("📈 Query Results")
                st.dataframe(st.session_state.query_result)
                
                # Attempt to generate a chart
                chart_type = get_chart_type(st.session_state.query_result)
                if chart_type:
                    plot_data(st.session_state.query_result, chart_type)
        else:
            # This case handles when the model returns "INVALID"
             st.warning("The model determined the question is not answerable by the database schema.")