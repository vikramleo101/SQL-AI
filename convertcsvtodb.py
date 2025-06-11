import argparse
import sqlite3
import pandas as pd
import os

def csv_to_sqlite(csv_dir, db_path):
    """
    Convert multiple CSV files from a directory to a single SQLite database
    
    Args:
        csv_dir (str): Path to directory containing input CSV files
        db_path (str): Path to output SQLite database file
    """
    conn = None # Initialize conn to None
    try:
        # Create SQLite database connection
        conn = sqlite3.connect(db_path)
        all_dfs = []
        
        for filename in os.listdir(csv_dir):
            if filename.endswith('.csv'):
                csv_path = os.path.join(csv_dir, filename)
                table_name = os.path.splitext(filename)[0] # Use filename without extension as table name
                
                # Read CSV file
                df = pd.read_csv(csv_path)
                
                # Prepend table_name to column names to ensure uniqueness in merged table
                df.columns = [f"{table_name}_{col}" for col in df.columns]

                # Write DataFrame to SQLite
                df.to_sql(table_name, conn, index=False, if_exists='replace')
                
                print(f"Successfully created table '{table_name}' with {len(df)} rows from {filename}")
                all_dfs.append(df)
        
        # Merge all data into a single DataFrame and create a new table
        if all_dfs:
            # Get all unique columns from all DataFrames
            all_columns = pd.Index([])
            for df in all_dfs:
                all_columns = all_columns.union(df.columns)
            
            # Reindex each DataFrame to have all columns, then concatenate
            merged_df = pd.concat([df.reindex(columns=all_columns) for df in all_dfs], ignore_index=True)
            
            # Write the merged DataFrame to a new table
            merged_df.to_sql('all_merged_data', conn, index=False, if_exists='replace')
            print(f"Successfully created 'all_merged_data' table with {len(merged_df)} rows containing all merged data.")

        # Add creation timestamp metadata for the whole database
        conn.execute(f"CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute(
            "INSERT OR REPLACE INTO metadata VALUES (?, ?)",
            ("creation_date", pd.Timestamp.now().isoformat())
        )
        
        print(f"All CSV files from {csv_dir} successfully merged into {db_path}")
        return True
        
    except Exception as e:
        print(f"Error merging CSVs to SQLite: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV files from a directory to a single SQLite database")
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing CSV files')
    parser.add_argument('--output', type=str, required=True, help='Output SQLite database path')
    
    args = parser.parse_args()
    
    csv_to_sqlite(
        csv_dir=args.input_dir,
        db_path=args.output
    )
