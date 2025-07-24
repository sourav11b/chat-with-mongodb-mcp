import asyncio
import os
import json
import sys # Import sys for error handling info
import uuid # For generating unique session IDs

# Correct import for mcp.client.stdio.stdio_client
from mcp import ClientSession, StdioServerParameters 
from mcp.client.stdio import stdio_client 

# Correct import for LangChain's message types, including ToolCall
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, ToolCall 
from dotenv import load_dotenv
from contextlib import AsyncExitStack
# REMOVED: from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI # NEW: Import ChatOpenAI

from langchain_mongodb import MongoDBChatMessageHistory

load_dotenv() # Load environment variables from .env file first
# REMOVED: openai_client = OpenAI()

MONGO_URI = os.getenv("MONGO_URI")
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME") # not used in current version

# Generate a new, unique session_id using UUID
session_id = str(uuid.uuid4()) 
print(f"Generated new chat session_id: {session_id}")

        
async def chat_loop():
    """Run an interactive chat loop"""
    print("\nMCP Client Started!")
    print("Type your queries or 'quit' to exit.")
    
    # Initialize LangChain's ChatOpenAI
    # You can set max_tokens here if you want a global limit for the LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0) # Added temperature for consistency, can be adjusted

    async with AsyncExitStack() as current_exit_stack:
        server_params = StdioServerParameters(
            command="npx",
            args=[ "-y",
            "mongodb-mcp-server",
            "--connectionString",
            MONGO_URI],
            env=None
        )

        stdio_transport = await current_exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport

        session = await current_exit_stack.enter_async_context(ClientSession(stdio, write))
        await session.initialize()

        # Initialize MongoDBChatMessageHistory ONCE here, before the while loop!
        chat_history_manager = MongoDBChatMessageHistory(
            connection_string=MONGO_URI,
            session_id=session_id, # Use the newly generated session_id
            database_name="historical_db",
            collection_name="chat_messages"
        )

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


        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break

                # Add user message to history
                chat_history_manager.add_user_message(query)

                # Flag to control the tool calling loop
                should_call_llm_again = True
                final_text = []

                while should_call_llm_again:
                    should_call_llm_again = False # Assume no more tool calls unless found

                    # Retrieve full message history (these are LangChain message types)
                    messages = chat_history_manager.messages
                    
                    # Convert MCP tools to LangChain's OpenAI format for tools
                    response = await session.list_tools()
                    langchain_tools = []
                    for tool in response.tools:
                        # For ChatOpenAI, we need to provide tools in the format expected by LangChain,
                        # which then converts them for the OpenAI API.
                        # This typically means a Pydantic model or a dictionary that can be converted.
                        # For now, let's stick to the dictionary format that ChatOpenAI understands for tools.
                        langchain_tools.append({
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.inputSchema
                            }
                        })
                    
                    # OpenAI API call using LangChain's ChatOpenAI.invoke
                    # Pass LangChain messages directly. ChatOpenAI handles the conversion.
                    llm_response = await llm.ainvoke(
                        input=messages, # Pass the list of LangChain messages directly
                        tools=langchain_tools,
                        tool_choice="auto" # This is a direct parameter for ainvoke with tools
                    )

                    # llm_response is already an AIMessage object from LangChain
                    response_message = llm_response 
                    
                    # Add AI message to history, including any tool calls it made
                    ai_message_content = response_message.content
                    ai_tool_calls = response_message.tool_calls # This is already a list of ToolCall objects
                    
                    if ai_tool_calls:
                        should_call_llm_again = True # AI wants to call a tool, so loop again

                    if response_message.content:
                        final_text.append(response_message.content)

                    # Handle tool calls if present
                    if response_message.tool_calls:
                        for tool_call in response_message.tool_calls:
                            tool_name = tool_call["name"] # Access name directly from ToolCall object
                            tool_args = tool_call["args"] # Access args (already a dict) directly from ToolCall object

                            try:
                                # Execute tool call
                                result = await session.call_tool(tool_name, tool_args) # tool_args is already a dict
                                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                                tool_output_content = str(result.content)
                                # print(f"\nTool {tool_name} output: {tool_output_content}")
                            except Exception as tool_e:
                                tool_output_content = f"Error executing tool {tool_name}: {str(tool_e)}"
                                print(f"\n{tool_output_content}")
                                
                            # Add tool message to history
                            
                            print("-----"+str(tool_call))
                            ai_tool_calls = []
                            ai_tool_calls.append(
                                ToolCall(id=tool_call["id"], name=tool_call["name"], args=tool_call["args"])
                            )
                            
                            chat_history_manager.add_ai_message( 
                                AIMessage(content=ai_message_content if ai_message_content is not None else "", tool_calls=ai_tool_calls)
                            )
                                
                            chat_history_manager.add_message( 
                                ToolMessage(
                                    content=tool_output_content,
                                    tool_call_id=tool_call["id"],
                                )
                            )
                    else:
                        # If no tool calls, add the AI's content-only message and exit the inner loop
                        '''
                        chat_history_manager.add_ai_message(
                            AIMessage(content=ai_message_content if ai_message_content is not None else "")
                        )
                        '''
                        should_call_llm_again = False
                    '''
                    if response_message.tool_calls and not response_message.content:
                         # If only tool calls and no content, we just add the tool call message
                        chat_history_manager.add_ai_message(
                            AIMessage(content="", tool_calls=response_message.tool_calls)
                        )
                    elif response_message.tool_calls and response_message.content:
                        chat_history_manager.add_ai_message(
                            AIMessage(content=response_message.content, tool_calls=response_message.tool_calls)
                        )
                    elif response_message.content and not response_message.tool_calls:
                        chat_history_manager.add_ai_message(
                            AIMessage(content=response_message.content)
                        )
                    '''


                response_output = "\n".join(final_text)
                print("\n" + response_output)

            except Exception as e:
                print(f"\nError: {str(e)}")
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(f"Error in {fname} at line {exc_tb.tb_lineno}")
                # Continue the loop even if an error occurs, allowing the user to type 'quit'
                # or try another query.

async def main():
    """Entry point for the client script."""
    await chat_loop()

if __name__ == "__main__":
    sys.set_int_handler = lambda x: None 
    asyncio.run(main())