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
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_mongodb import MongoDBChatMessageHistory

load_dotenv() # Load environment variables from .env file first
openai_client = OpenAI()

# Generate a new, unique session_id using UUID
session_id = str(uuid.uuid4()) 
print(f"Generated new chat session_id: {session_id}")


        
async def chat_loop():
    """Run an interactive chat loop"""
    print("\nMCP Client Started!")
    print("Type your queries or 'quit' to exit.")
    
    

    async with AsyncExitStack() as current_exit_stack:
        server_params = StdioServerParameters(
            command="npx",
            args=[ "-y",
            "mongodb-mcp-server",
            "--connectionString",
            MONGO_CONNECTION_STRING],
            env=None
        )

        stdio_transport = await current_exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport

        session = await current_exit_stack.enter_async_context(ClientSession(stdio, write))
        await session.initialize()

        # Initialize MongoDBChatMessageHistory ONCE here, before the while loop!
        chat_history_manager = MongoDBChatMessageHistory(
            connection_string=MONGO_CONNECTION_STRING,
            session_id=session_id, # Use the newly generated session_id
            database_name="historical_db",
            collection_name="chat_messages"
        )

        # Helper function to convert LangChain messages to OpenAI API format
        # This function now correctly handles the sequencing for 'tool' messages
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
                                    "id": tc["id"], # Access id directly from ToolCall object
                                    "type": "function",
                                    "function": {
                                        "name": tc["name"], # Access name directly from ToolCall object
                                        "arguments": json.dumps(tc["args"]) # Access args directly from ToolCall object
                                    }
                                })
                            except Exception as e:
                                print(f"WARNING: Failed to format a tool call in to_openai_api_messages. Skipping. Error: {e}, ToolCall: {tc}")
                                continue 
                        
                        if formatted_tool_calls: # If we successfully formatted any tool calls
                            oai_msg["tool_calls"] = formatted_tool_calls
                            oai_msg["content"] = msg.content if msg.content is not None else "" 
                        else: # No valid tool calls were formatted, or original msg.tool_calls was empty
                            oai_msg["content"] = msg.content 

                    else: # Original msg.tool_calls was empty or None (text-only AI response)
                        oai_msg["content"] = msg.content 
                        
                    oai_messages.append(oai_msg)

                elif isinstance(msg, ToolMessage):
                    # --- CRITICAL FIX HERE ---
                    # Only add ToolMessage if the IMMEDIATELY preceding message in the *currently building* oai_messages
                    # list is an assistant message that specified tool_calls.
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

                    # Retrieve full message history
                    messages = chat_history_manager.messages
                    
                    # Convert LangChain messages to OpenAI API format
                    openai_messages = to_openai_api_messages(messages)
                    
                    response = await session.list_tools()
                    available_tools = [{
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema
                        }
                    } for tool in response.tools]
                    
                    # OpenAI API call with history
                    llm_response = openai_client.chat.completions.create(
                        model="gpt-4o",
                        max_tokens=1000,
                        messages=openai_messages, 
                        tools=available_tools,
                        tool_choice="auto"
                    )

                    response_message = llm_response.choices[0].message
                    
                    # Add AI message to history, including any tool calls it made
                    ai_message_content = response_message.content
                    ai_tool_calls = []
                    if response_message.tool_calls:
                        for tc in response_message.tool_calls:
                            try:
                                tc_args_dict = json.loads(tc.function.arguments) 
                            except json.JSONDecodeError:
                                print(f"Warning: Could not decode tool arguments for {tc.function.name}: {tc.function.arguments}")
                                tc_args_dict = {}
                            
                            ai_tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, args=tc_args_dict)) 
                        should_call_llm_again = True # AI wants to call a tool, so loop again

                    

                    if response_message.content:
                        final_text.append(response_message.content)

                    # Handle tool calls if present
                    if response_message.tool_calls:
                        for tool_call in response_message.tool_calls:
                            tool_name = tool_call.function.name
                            tool_args = tool_call.function.arguments

                            try:
                                parsed_tool_args = json.loads(tool_args)
                            except json.JSONDecodeError:
                                print(f"Warning: Could not decode tool arguments for {tool_name}: {tool_args}")
                                parsed_tool_args = {}

                            try:
                                # Execute tool call
                                result = await session.call_tool(tool_name, parsed_tool_args)
                                final_text.append(f"[Calling tool {tool_name} with args {parsed_tool_args}]")
                                tool_output_content = str(result.content)
                                # print(f"\nTool {tool_name} output: {tool_output_content}")
                            except Exception as tool_e:
                                tool_output_content = f"Error executing tool {tool_name}: {str(tool_e)}"
                                print(f"\n{tool_output_content}")
                                # Do not set should_call_llm_again to False here,
                                # as the LLM might still want to try other tools or respond.
                                # The error is handled and reported.
                                
                            # Add tool message to history
                            ai_tool_calls = []
                            tc_args_dict = json.loads(tool_call.function.arguments) 
                            ai_tool_calls.append(
                                ToolCall(id=tool_call.id, name=tool_call.function.name, args=tc_args_dict)
                            )
                            chat_history_manager.add_ai_message( 
                                AIMessage(content=ai_message_content if ai_message_content is not None else "", tool_calls=ai_tool_calls)
                            )
                            
                            chat_history_manager.add_message( 
                                ToolMessage(
                                    content=tool_output_content,
                                    tool_call_id=tool_call.id,
                                )
                            )
                    else:
                        # If no tool calls, exit the inner loop
                        should_call_llm_again = False

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
