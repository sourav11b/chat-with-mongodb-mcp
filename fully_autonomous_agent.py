# Import necessary libraries.
# asyncio: For asynchronous programming, crucial for the chat loop.
import asyncio
# os: To interact with the operating system, particularly to get environment variables.
import os
# json: For working with JSON data, specifically for formatting tool outputs.
import json
# sys: Provides access to system-specific parameters and functions, used for error handling.
import sys
# uuid: To generate unique identifiers for chat sessions.
import uuid
# contextlib: Provides utilities for with-statement contexts, like AsyncExitStack.
from contextlib import AsyncExitStack
# datetime and timezone: For working with dates and times, used by the 'get_utc_time' tool.
from datetime import datetime, timezone

# pymongo: The official Python driver for MongoDB.
from pymongo import MongoClient
# VoyageAIEmbeddings: A LangChain integration for creating text embeddings using Voyage AI.
from langchain_voyageai import VoyageAIEmbeddings

# langchain_core: Contains core classes for LangChain, like messages and tool calls.
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, ToolCall
# langchain_core.tools.tool: Decorator to easily create LangChain tools.
from langchain_core.tools import tool
# AzureChatOpenAI: LangChain's class for interacting with Azure's OpenAI service.
from langchain_openai.chat_models import AzureChatOpenAI 
# MongoDBChatMessageHistory: A LangChain class for storing chat history in MongoDB.
from langchain_mongodb import MongoDBChatMessageHistory
# dotenv: To load environment variables from a .env file.
from dotenv import load_dotenv

# mcp (MongoDB Command Protocol): Libraries for interacting with the MongoDB Command Server.
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# Load environment variables from a .env file.
# This makes it easy to manage sensitive information like API keys and connection strings.
load_dotenv()

# --- Configuration from Environment Variables ---
# These variables are loaded from the .env file and are essential for connecting to
# MongoDB Atlas and Azure OpenAI.
ATLAS_URI = os.getenv("ATLAS_URI")
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME")
ATLAS_DB_NAME = os.getenv("ATLAS_DB_NAME")
ATLAS_COLLECTION_NAME_MANUALS = os.getenv("ATLAS_COLLECTION_NAME_MANUALS")

# --- Azure OpenAI Configuration ---
# Variables for connecting to the Azure OpenAI service.
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") 


# --- MongoDB Client Initialization ---
# Initialize the MongoClient to connect to the MongoDB Atlas cluster.
mongo_client = MongoClient(ATLAS_URI)
# Select the specific database and collection to be used for the searches.
collection = mongo_client[ATLAS_DB_NAME][ATLAS_COLLECTION_NAME_MANUALS]


# --- Voyage AI Client Initialization ---
# Initialize the VoyageAIEmbeddings client with a specific model.
voyage_client = VoyageAIEmbeddings( model="voyage-3-lite")

# Define a function to generate embeddings.
# This function is used by the vector and fusion search tools to convert text into numerical vectors.
def get_embedding(data, input_type = "document"):
  embeddings = voyage_client.embed_query(data )
  return embeddings
  
  
# Define a tool using the @tool decorator.
@tool
def resolve_alert( alert : str) -> str:
    """performs steps needed to rectify alert condition sent as input"""
    print(f"action performed to remedy alert {alert}")
    return f"action performed to remedy alert {alert}"
  
# Define a tool using the @tool decorator.
@tool
def get_utc_time() -> str:
    """Returns the current UTC date and time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


# Define the vector search tool.
# This tool performs a vector search on the MongoDB collection using a user's input.
@tool
def vector_search_tool(user_input: str) -> str:
    """
    Search MongoDB collection using vector search or semantic search for given user_input string. 
    The results contains steps needed to fix the issue described by user_input
    
    Args:
        user_input (str): The name alert conditions
    Returns:
        str: string representation of a list of documents from mongodb collection which contains possible   solutions for the input alert condition
    """
    # First, get the embedding for the user's input query.
    query_embedding = get_embedding(user_input)

    try :
        # Define the MongoDB aggregation pipeline for vector search.
        pipeline = [
          {
             "$vectorSearch": {
              "index": ATLAS_VECTOR_SEARCH_INDEX_NAME,
              "queryVector": query_embedding,
              "path": "embedding",
              "exact": True,
              "limit": 5
             }
          },
          # Project the results to exclude the _id field for cleaner output.
          { "$project" : {
                            "_id" : 0
                        }
          }
        ]
        results = collection.aggregate(pipeline)
    except Exception as e:
        print(f"Caught a general exception: {e}")
        # If an error occurs, return a JSON string with the error message.
        return json.dumps({"error": str(e)}) 

    array_of_results = []
    # Iterate through the results and append each document to a list.
    for doc in results:
        array_of_results.append(doc)

    # Convert the list of dictionaries (MongoDB documents) to a JSON formatted string.
    return json.dumps(array_of_results, indent=2) # Using indent for pretty printing the JSON

# Define the text search tool.
# This tool performs a traditional keyword-based text search.
@tool
def text_search_tool(user_input: str) -> str:
    """
    Search MongoDB collection using text search  for given user_input string. 
    The results contains steps needed to fix the issue described by user_input
    
    Args:
        user_input (str): The name alert conditions
    Returns:
        str: string representation of a list of documents from mongodb collection which contains possible   solutions for the input alert condition
    """
    try :
        # Define the MongoDB aggregation pipeline for text search.
        results = collection.aggregate([
          {
            '$search': 
             {
                'index': "default_search",
                'text': {
                  'query': user_input,
                  'path': {
                        'wildcard': "*"
                    }
                }
             }
          }, 
          {
            '$limit': 5
          }, 
          {
            '$project': {
               '_id': 0
            }
          }
        ])
    except Exception as e:
        print(f"Caught a general exception: {e}")
        return json.dumps({"error": str(e)}) # Return an error message as JSON

    array_of_results = []
    for doc in results:
        array_of_results.append(doc)

    return json.dumps(array_of_results, indent=2) # Using indent for pretty printing the JSON
    
# Define the fusion search tool.
# This tool combines both vector and text search using rank fusion to get more relevant results.
@tool
def fusion_search_tool(user_input: str) -> str:
    """
    Search MongoDB collection using fusion search  for given user_input string. 
    The results contains steps needed to fix the issue described by user_input
    
    Args:
        user_input (str): The name alert conditions
    Returns:
        str: string representation of a list of documents from mongodb collection which contains possible   solutions for the input alert condition
    """
    # Get the embedding for the user input, required for the vector search part of the pipeline.
    query_embedding = get_embedding(user_input)
    try :
        # Define the MongoDB aggregation pipeline for rank fusion search.
        results = collection.aggregate(
            [
               {
                  "$rankFusion": {
                     'input': {
                        'pipelines': {
                           # Pipeline for the vector search.
                           "searchOne": [
                              {
                                 "$vectorSearch": {
                                  "index": ATLAS_VECTOR_SEARCH_INDEX_NAME,
                                  "queryVector": query_embedding,
                                  "path": "embedding",
                                  "exact": True,
                                  "limit": 5
                                 }
                              }
                           ],
                           # Pipeline for the text search.
                           "searchTwo": [
                              {
                                 '$search': {
                                        'text': {
                                          'query': user_input,
                                          'path': {
                                                'wildcard': "*"
                                            }
                                        }
                                     }
                              },
                              { "$limit": 5 }
                           ],
                        }
                     }
                  }
               },
               { '$limit': 5 }
            ] 
        )
    except Exception as e:
        print(f"Caught a general exception: {e}")
        return json.dumps({"error": str(e)}) # Return an error message as JSON

    array_of_results = []
    for doc in results:
        array_of_results.append(doc)

    return json.dumps(array_of_results, indent=2) # Using indent for pretty printing the JSON
 
# This function is a helper to convert LangChain messages into the format expected by the
# OpenAI API. This is important for ensuring the model correctly interprets the chat history,
# especially when it involves tool calls.
def to_openai_api_messages(lc_messages):
            oai_messages = []
            for msg in lc_messages:
                if isinstance(msg, HumanMessage):
                    oai_messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    oai_msg = {"role": "assistant"}
                    
                    formatted_tool_calls = []
                    if msg.tool_calls: # Check if the LangChain AIMessage has tool_calls listed
                        for tc in msg.tool_calls:
                            try:
                                formatted_tool_calls.append({
                                    "id": tc.id, # Access id directly from ToolCall object
                                    "type": "function",
                                    "function": {
                                        "name": tc.name, # Access name directly from ToolCall object
                                        "arguments": json.dumps(tc.args) # Access args directly from ToolCall object
                                    }
                                })
                            except Exception as e:
                                print(f"WARNING: Failed to format a tool call in to_openai_api_messages. Skipping. Error: {e}, ToolCall: {tc}")
                                continue 
                        
                        if formatted_tool_calls: 
                            oai_msg["tool_calls"] = formatted_tool_calls
                            oai_msg["content"] = msg.content if msg.content is not None else "" 
                        else: 
                            oai_msg["content"] = msg.content 

                    else: 
                        oai_msg["content"] = msg.content 
                        
                    oai_messages.append(oai_msg)

                elif isinstance(msg, ToolMessage):
            
                    if (oai_messages and 
                        oai_messages[-1].get("role") == "assistant" and 
                        "tool_calls" in oai_messages[-1] and 
                        oai_messages[-1]["tool_calls"] # Ensure tool_calls array is not empty
                    ):
                        oai_messages.append({
                            "role": "tool",
                            "tool_call_id": msg.tool_call_id,
                            "content": msg.content
                        })
                    else:
                        # This ToolMessage is out of sequence for OpenAI. Skip it to prevent error.
                        print(f"WARNING: Skipping ToolMessage (ID: {msg.tool_call_id}) because it's not preceded by a valid assistant tool_calls message for OpenAI API.")
                        continue # DO NOT ADD THIS MESSAGE TO THE OAI_MESSAGES LIST
            return oai_messages

# The main asynchronous function that runs the chat loop.
async def chat_loop():
    """Run a single query automatically without user input."""
    print("\nMCP Client Started!")

    # Initialize LangChain's AzureChatOpenAI with the loaded configuration.
    llm = AzureChatOpenAI(
        openai_api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        temperature=0
    )


    # Use AsyncExitStack for proper cleanup of asynchronous contexts.
    async with AsyncExitStack() as current_exit_stack:
        # Define the parameters for the MongoDB Command Protocol (MCP) server.
        server_params = StdioServerParameters(
            command="npx",
            args=[ "-y",
            "mongodb-mcp-server",
            "--connectionString",
            ATLAS_URI],
            env=None
        )

        # Connect to the MCP server.
        stdio_transport = await current_exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport

        # Start a client session with the MCP server.
        session = await current_exit_stack.enter_async_context(ClientSession(stdio, write))
        await session.initialize()

        # Initialize MongoDBChatMessageHistory ONCE here, before the while loop!
        # This will store the chat history in a MongoDB collection.
        chat_history_manager = MongoDBChatMessageHistory(
            connection_string=ATLAS_URI,
            session_id=str(uuid.uuid4()), # Use the newly generated session_id
            database_name="historical_db",
            collection_name="chat_messages"
        )

        # Get the list of tools from the MCP server.
        response = await session.list_tools()
        langchain_tools = []
        # Convert the MCP tools into a format that LangChain and Azure OpenAI can use.
        for tool in response.tools:
            langchain_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            })
                    
        # Manually add the custom-defined tools (vector, text, fusion, and get_utc_time)
        # to the list of tools the model can use.
        langchain_tools.append({
            "type": "function",
            "function": {
                "name": vector_search_tool.name,
                "description": vector_search_tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_input": {"type": "string", "description": "The name alert conditions"}
                    },
                    "required": ["user_input"]
                }
            }
        })
                    
        langchain_tools.append({
            "type": "function",
            "function": {
                "name": text_search_tool.name,
                "description": text_search_tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_input": {"type": "string", "description": "The name alert conditions"}
                    },
                    "required": ["user_input"]
                }
            }
        })
                    
        langchain_tools.append({
            "type": "function",
            "function": {
                "name": fusion_search_tool.name,
                "description": fusion_search_tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_input": {"type": "string", "description": "The name alert conditions"}
                    },
                    "required": ["user_input"]
                }
            }
        })
        langchain_tools.append({
            "type": "function",
            "function": {
                "name": get_utc_time.name,
                "description": get_utc_time.description,
                "parameters": {}
            }
        })
        langchain_tools.append({
            "type": "function",
            "function": {
                "name": resolve_alert.name,
                "description": resolve_alert.description,
                "parameters": {}
            }
        })
        

        # --- Automated Queries Section ---
        queries = [
            " Please access the realtime network logs collection in the network monitoring database and, after identifying the correct UTC timestamp field by examining its schema, retrieve all event descriptions inserted within the last 20 seconds; then, summarize these descriptions, pinpointing any alert conditions where the severity is 4 or greater, and finally,if alerts were founds, leverage vector search tools to find and present possible solutions for all identified alerts.perform actions needed to rectify the alerts.add an entry to a findings collection with identified alerts and resultions. If you find alerts print the id of the records in findings collection corresponding to them.wait for 20 seconds after  execution is complete"
        ]
        
        query = "note the timestamp. access the realtime network logs collection in the network monitoring database and, after identifying the correct UTC timestamp field by examining its schema, retrieve all event descriptions inserted within the last 20 seconds; then, summarize these descriptions, pinpointing any alert conditions where the severity is 4 or greater, and finally,if alerts were founds, leverage vector search tools to find and present possible solutions for all identified alerts.add an entry to a findings collection with identified alerts and resultions. If you find alerts print the id of the records in findings collection corresponding to them.if no laerts were found print a statement as well.wait for atleast 20 seconds from the timestamp you noted at the beginning"

        while  True:
            print(f"\nRunning automated query: '{query}'")

            try:
                # Add the user's message to the chat history.
                chat_history_manager.add_user_message(query)

                # A flag to control the loop for multi-step tool calls.
                should_call_llm_again = True
                final_text = []

                # This inner loop handles multi-turn conversations and tool calls.
                while should_call_llm_again:
                    should_call_llm_again = False # Assume no more tool calls unless found

                    # Retrieve the entire message history.
                    messages = chat_history_manager.messages
                    
                    # Invoke the LLM with the message history and the available tools.
                    llm_response = await llm.ainvoke(
                        input=messages, # Pass the list of LangChain messages directly
                        tools=langchain_tools,
                        tool_choice="auto" # Let the model decide which tool to use, if any.
                    )

                    # Get the AI's response message object.
                    response_message = llm_response 
                    
                    # Get the content and any tool calls from the AI's response.
                    ai_message_content = response_message.content
                    ai_tool_calls = response_message.tool_calls 
                    
                    if ai_tool_calls:
                        should_call_llm_again = True # If tools are called, loop again after execution.

                    if response_message.content:
                        # Append any text content from the AI's response.
                        final_text.append(response_message.content)
                    
                    # Handle tool calls if they exist in the AI's response.
                    if response_message.tool_calls:
                        
                        for tool_call in response_message.tool_calls:
                            tool_name = tool_call["name"] 
                            tool_args = tool_call["args"] 
                            # Check if the tool is one of the manually defined ones.
                            if tool_name not in ["vector_search_tool","get_utc_time","text_search_tool","fusion_search_tool"]: 
                                try:
                                    # Execute tool calls for tools from the MCP server.
                                    result = await session.call_tool(tool_name, tool_args) 
                                    final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                                    tool_output_content = str(result.content)
                                except Exception as tool_e:
                                    tool_output_content = f"Error executing tool {tool_name}: {str(tool_e)}"
                                    print(f"\n{tool_output_content}")
                            else :
                                # Execute the custom Python-based tools.
                                try: 
                                    if tool_name == "vector_search_tool" :
                                        result = vector_search_tool.invoke(tool_args)  
                                    elif tool_name == "text_search_tool" :
                                        result = text_search_tool.invoke(tool_args)  
                                    elif tool_name == "fusion_search_tool" :
                                        result = fusion_search_tool.invoke(tool_args) 
                                    elif tool_name == "resolve_alert" :
                                        result = resolve_alert.invoke(tool_args)
                                    
                                    else :                                        
                                        result = get_utc_time.invoke(tool_args)
                                    final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                                    tool_output_content = str(result) 
                                except Exception as tool_e:
                                    tool_output_content = f"Error executing tool {tool_name}: {str(tool_e)}"
                                    print(f"\n{tool_output_content}")
                            
                            # Add the AI message with the tool call to the history.
                            ai_tool_calls = []
                            ai_tool_calls.append(
                                ToolCall(id=tool_call["id"], name=tool_call["name"], args=tool_call["args"])
                            )
                            
                            chat_history_manager.add_ai_message( 
                                AIMessage(content=ai_message_content if ai_message_content is not None else "", tool_calls=ai_tool_calls)
                            )
                                
                            # Add the result of the tool execution as a ToolMessage to the history.
                            chat_history_manager.add_message( 
                                ToolMessage(
                                    content=tool_output_content,
                                    tool_call_id=tool_call["id"],
                                )
                            )
                    else:
                        # If no tool calls, add the AI's content-only message and exit the inner loop.
                        chat_history_manager.add_ai_message(
                            AIMessage(content=ai_message_content if ai_message_content is not None else "")
                        )
                        should_call_llm_again = False
                    
                # Join all the parts of the final response and print it.
                response_output = "\n".join(final_text)
                print("\n" + response_output)

            except Exception as e:
                # Basic error handling to catch and print exceptions.
                print(f"\nError: {str(e)}")
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(f"Error in {fname} at line {exc_tb.tb_lineno}")
        
        print("\nAll automated queries have been processed. Exiting.")
        
# The entry point of the script.
async def main():
    """Entry point for the client script."""
    await chat_loop()

if __name__ == "__main__":
    # Run the main asynchronous function.
    asyncio.run(main())