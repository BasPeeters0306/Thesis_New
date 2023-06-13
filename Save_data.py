import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import engine
from sqlalchemy.engine import URL
import os

# This function writes a dataframe to SQL
def save_to_SQL(df, sql_table_name,):
    
    # Remove infinity values to prevent errors
    df.replace({np.inf: np.nan, -np.inf: np.nan}, inplace=True)

    db_name = "Bas.Thesis.Local"
    server_name = "DESKTOP-JF8CK5U"  # 10.0.1.6\MAMSTUDIO_DEV

    # Makes connection to write to SQL
    conn_string = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=" + server_name + ";DATABASE=" + db_name + ";UID=sa;PWD=sa"
    conn_url = URL.create("mssql+pyodbc", query={"odbc_connect": conn_string})
    db_engine = engine.create_engine(conn_url, fast_executemany=True, connect_args={'connect_timeout': 10}, echo=False)

    # Writes the dataframe to SQL
    df.to_sql(con=db_engine, name=sql_table_name, if_exists="replace", index=False, chunksize=2000)


# This function writes a dataframe to a csv file
def save_to_csv(df, file_name):

    directory = r"C:\Users\BasPeeters\OneDrive - FactorOrange.capital\Master Thesis\Dataframes and output"
    full_path = os.path.join(directory, f"{file_name}.csv")

    # Save df to csv
    df.to_csv(full_path, index=False)

    print(f"Dataframe saved to {full_path}")

