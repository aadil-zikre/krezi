# INTRODUCTION 

1. This is a simple util that takes username, password and server to connect to any mssql database. 
2. After the connection is successful, you can run queries by writing queries as would in any sql client but passing the queries as strings to the query function. Query function is a object attribute of the pymssql_dao class. 
3. You can also dry run your queries by calling query builder. It will only show you the final query to be executed without actually running it. 
4. You can change schema by calling change schema method

# THINGS TO BE ADDED

1. Logger Support 
2. Fetching Data in Batches
3. max rows to fetch from a database 
4. Error Handling in case connection fails 
5. Connection Retries 