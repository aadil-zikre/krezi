import pandas as pd
import sqlalchemy

class mssql_dao:
    """
    Class to run Queries on MSSQL DB Schema Provided
    """
    def __init__(self, conn_dict, schema = 'vbay'):
        self.conn_dict = conn_dict 
        self.schema = schema 
        self.engine = None
        self.cursor = None
    
    def init_engine(self):
        cnxn_str = f"mssql+pymssql://{self.conn_dict['user']}:{self.conn_dict['password']}@{self.conn_dict['server']}/{self.schema}"
        print(f"Trying to establish a connection with connection string \n{cnxn_str}")
        self.engine = sqlalchemy.create_engine(cnxn_str)
        
    def init_cursor(self):
        if not self.engine:
            self.init_engine()
        print("Initializing Cursor")
        self.cursor = self.engine.raw_connection().cursor(as_dict=True)
    
    def change_schema(self,schema):
        print(f"Changing Schema from {self.schema} to {schema}")
        self.schema = schema
        self.init_engine()
        self.init_cursor()
        
    @staticmethod
    def query_builder(query_str, params):
        """
        A simple function that returns the final query that can be executed with actually executing the query
        Return: String

        Note: This is a static method. It is safe to use from outside the class
        """
        if not isinstance(query_str, str): 
            raise TypeError(f"query should be of type str but {type(query_str).__name__} was passed!")
        if not isinstance(params, dict): 
            raise TypeError(f"params should be of type dict but {type(params).__name__} was passed!") 

        for k in params:
            query_str = query_str.replace(f":{k}", f"%({k})s")
        
        return query_str
    
    def query(self, query_str, params = {}):
        """
        Inputs:     
            query_str : str : query as is written in sql client
            params : dict : default empty : pass parameters in a dictionary and refer to them in query_str using a ':'
        Returns:
            DataFrame
        Note: As a good practice, only call this function from outside
        """
        final_query = self.query_builder(query_str, params)
        if not self.cursor:
            self.init_cursor()
        if not self.cursor.connection:
            self.init_cursor()
        self.cursor.execute(final_query, params)
        rows = self.cursor.fetchall()
        res_df = pd.DataFrame(rows)
        res_df.columns = res_df.columns.str.replace(" ","_").str.upper() # uppercasing all columns as well as replacing any whitespaces with underscores
        return res_df