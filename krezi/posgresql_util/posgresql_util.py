# from az_common_funcs import *

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

class pgdao:
    def __init__(self, conn_dict, default_db = 'airbnb'):
        self.conn_dict = conn_dict
        self.database = default_db if conn_dict.get('database') is None else conn_dict.get('database')
        self.host = conn_dict.get('host')
        self.port = conn_dict.get('port')
        self.user = conn_dict.get('user')
        self.password = conn_dict.get('password')
        self.conn = None
        self.cursor = None
    
    def _init_connection(self):
        self.conn = psycopg2.connect(
                                host = self.host,
                                port = self.port,
                                database = self.database,
                                user = self.user,
                                password = self.password)   
        self.conn.set_session(readonly=True)  
        
        
    def _init_cursor(self):
        if not self.conn or self.conn.closed!=0:
            self._init_connection()
        self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
    
    def change_database(self,database):
        self.database = database
        self._init_connection()

    def _query_builder(self, query_str, params):
        if not isinstance(query_str, str): 
            raise TypeError(f"query should be of type str but {type(query_str).__name__} was passed!")
        if not isinstance(params, dict): 
            raise TypeError(f"params should be of type dict but {type(params).__name__} was passed!") 

        for k in params:
            query_str = query_str.replace(f":{k}", f"%({k})s")
        
        return query_str
    
    def query(self, query_str, params = {}, dry_run=False, show_query=True):
        final_query = self._query_builder(query_str, params)
        if not self.cursor or self.cursor.closed != 0:
            self._init_cursor()
        self.conn.rollback()
        if dry_run:
            print(f"Dry Run Query : \n{self.cursor.mogrify(final_query, params)!r}")
        else:
            if show_query:
                print(f"Executing Query : \n{self.cursor.mogrify(final_query, params)!r}")
            self.cursor.execute(final_query, params)
            rows = self.cursor.fetchall()
            res_df = pd.DataFrame(rows)
            res_df.columns = res_df.columns.str.upper()
            self.conn.rollback()
            return res_df
    
    def query_fetch_in_batches(self, query_str, params = {}, dry_run=False, batch_size=100_000, max_size=500_000):
        final_query = self._query_builder(query_str, params)
        if not self.cursor or self.cursor.closed != 0:
            self._init_cursor()
        self.conn.rollback()
        if dry_run:
            print(f"Dry Run Query : \n{self.cursor.mogrify(final_query, params)!r}")
        else:
            print(f"Executing Query : \n{self.cursor.mogrify(final_query, params)!r}")
            self.cursor.execute(final_query, params)
            df_list = []
            row_count = min(self.cursor.rowcount, max_size)
            print(f"Rows found :: {self.cursor.rowcount}, Rows to retrieve :: {row_count}")
            while self.cursor.rownumber < row_count:
                fetch_size = min(batch_size, row_count-self.cursor.rownumber)
                print(f"Fetching next {fetch_size:_} rows")
                df_list.append(pd.DataFrame(self.cursor.fetchmany(fetch_size)))
            self.conn.rollback()
            return pd.concat(df_list, ignore_index=True)
    

            