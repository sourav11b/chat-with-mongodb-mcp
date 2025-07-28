# chat-with-mongodb-mcp
create a .env file and populate the keys

```
OPENAI_API_KEY = ""
VOYAGE_API_KEY =""
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index_embedding"
ATLAS_URI = ""  
ATLAS_DB_NAME = "network_monitoring"
ATLAS_COLLECTION_NAME = "realtime_network_logs"
ATLAS_COLLECTION_NAME_MANUALS = 'manuals_collection'
```

1. python -m venv mongo-mcp-chat
1. mongo-mcp-chat\Scripts\activate.bat ( this for Windows, change for other OS)
1. pip install -r requirements.txt

# chunk, create embeddings and upload pdf manual

```
python upload_manual.py
```
# start sending time series network logs

```
python send_data_to_ts.py
```

# start chat bot with memory and MCP tools 
```
python mcp_client.py
```
# sample prompt

```
Please access the realtime_network_logs table in the network monitoring database and, after identifying the correct UTC timestamp field by examining its schema, retrieve event descriptions inserted within the last 20 seconds; then, summarize these descriptions, pinpointing any alert conditions where the severity is 4 or greater, and finally, leverage vector search tools to find and present possible solutions for all identified alerts.
```

# sample output

```
Query:  Please access the realtime_network_logs table in the network monitoring database and, after identifying the correct UTC timestamp field by examining its schema, retrieve event descriptions inserted within the last 20 seconds; then, summarize these descriptions, pinpointing any alert conditions where the severity is 4 or greater, and finally, leverage vector search tools to find and present possible solutions for all identified alerts.
getting embeddings :

[Calling tool collection-schema with args {'database': 'network_monitoring', 'collection': 'realtime_network_logs'}]
[Calling tool get_utc_time with args {}]
[Calling tool find with args {'database': 'network_monitoring', 'collection': 'realtime_network_logs', 'filter': {'fieldtime': {'$gte': {'$date': '2025-07-28T18:46:23.286106+00:00'}}}, 'projection': {'eventdescription': 1, 'severity': 1, '_id': 0}}]
[Calling tool vector_search_tool with args {'user_input': 'Critical RF Module failure detected, impacting signal transmission and reception across multiple sectors, requiring immediate attention to restore full service capability.'}]
### Summary of Event Descriptions with Severity 4 or Greater

1. **Event Description**: Critical RF Module failure detected, impacting signal transmission and reception across multiple sectors, requiring immediate attention to restore full service capability.
   - **Severity**: 5
   - **Occurrences**: Multiple

### Possible Solutions for Identified Alerts

1. **RF Module Failure**:
   - **Description**: An active alarm indicating a malfunction or failure within one or more RF modules, which are often integrated into or connected to the Remote Radio Unit (RRU). This can lead to service degradation or complete outage for affected sectors, as the module is responsible for signal generation and reception.
   - **Probable Causes**:
     - Internal hardware fault within the RF module.
     - Missing or faulty connections to the module, including power, fiber, or RF jumpers.
     - Insufficient or unstable power supply to the module.
   - **Recommended Actions**:
     - Conduct a thorough visual inspection of the RF module and all its connections, including power cables, fiber optic cables, and RF jumpers. Look for any signs of physical damage, loose connections, or improper seating.
     - If the system supports it, attempt a unit block/unblock or a soft reset of the affected RF module.
     - Ensure stable and sufficient power supply to the module.

These solutions are derived from the manual and are aimed at addressing the critical RF module failures detected in the network logs.
```

# chat history

chat history per session stored here

```
            database_name="historical_db",
            collection_name="chat_messages"
```
