from pymongo import MongoClient

uri = "mongodb://<user>:<password>@localhost:27017/<db_name>"

client = MongoClient(uri)

''' 
Configure a new user for the db using mongosh before doing the next steps. If the user does not have read write permissions
the next part of the code will throw authentication error. 
'''

db = client['db_name']

collection = db['<collection_name>']


# Function to insert a document into a collection
def insert_document(collection, document):
    result = collection.insert_one(document)
    print("Document inserted with id:", result.inserted_id)

# Function to find documents in a collection
def find_documents(collection, query):
    documents = collection.find(query)
    for document in documents:
        print(document)

# Function to update a document in a collection
def update_document(collection, query, new_values):
    result = collection.update_one(query, {"$set": new_values})
    print("Modified:", result.modified_count)

# Function to delete a document from a collection
def delete_document(collection, query):
    result = collection.delete_one(query)
    print("Deleted:", result.deleted_count)

# Example usage
if __name__ == "__main__":
    pass

    # # Get the desired collection
    # collection = db['your_collection_name']

    # # Example document
    # document = {
    #     "name": "John Doe",
    #     "age": 30,
    #     "email": "john.doe@example.com"
    # }

    # # Insert a document
    # insert_document(collection, document)

    # # Find documents
    # find_documents(collection, {"name": "John Doe"})

    # # Update a document
    # update_document(collection, {"name": "John Doe"}, {"age": 35})

    # # Find and print updated document
    # find_documents(collection, {"name": "John Doe"})

    # # Delete a document
    # delete_document(collection, {"name": "John Doe"})
