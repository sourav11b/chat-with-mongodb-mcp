# dependency - tested with following software on Ubuntu

```
(demo) ubuntu@ip-:~$ lsb_release -a
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 24.04.3 LTS
Release:        24.04
Codename:       noble


(demo) ubuntu@ip-:~$ python --version
Python 3.12.3
(demo) ubuntu@ip-:~$ node -v
v24.12.0
(demo) ubuntu@ip-:~$ npm -v
11.6.2
```

# chat-with-mongodb-mcp
create a .env file and populate the keys

create Azure OpenAI Key

create Atlas Cluster

```
VOYAGE_API_KEY = ""
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index_embedding"
ATLAS_URI = ""  
ATLAS_DB_NAME = "network_monitoring"
ATLAS_COLLECTION_NAME = "realtime_network_logs"
ATLAS_COLLECTION_NAME_MANUALS = "manuals_collection"
ATLAS_HISTORY_DB_NAME = "webapp_history_db"
ATLAS_HISTORY_COLLECTION_NAME = "webapp_history_coll"
ATLAS_API_CLIENT_ID = ""
ATLAS_API_CLIENT_SECRET = ""
ATLAS_API_CLUSTER_NAME = "test-asp"

AZURE_OPENAI_API_KEY = ""
AZURE_OPENAI_ENDPOINT = ""
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
AZURE_OPENAI_DEPLOYMENT_NAME = ""
```

Install Requirements

```

1. python -m venv mongo-mcp-chat
1. mongo-mcp-chat\Scripts\activate.bat ( this for Windows, change for other OS source mongo-mcp-chat/bin/activate)
1. pip install -r requirements.txt
```

# chunk, create embeddings and upload pdf manual

```
python upload_pdf_manual.py

```


# start chat bot with memory and MCP tools 

For Azure OpenAI LLMs

```
streamlit run streamlit_app
```


control the number of errors per batch, batch  size, number of towers etc using these params in the file , ERRORS_PER_BATCH should be less than DOCUMENTS_PER_SEND

```
NUM_TOWERS = 10  # Number of towers sending data, each will get its own thread
DOCUMENTS_PER_SEND = 50  # Number of documents each tower sends per interval
SEND_INTERVAL_SECONDS = 5
ERRORS_PER_BATCH = 40  # Number of error documents to inject per batch (must be < DOCUMENTS_PER_SEND)
```
# Tools 

```
1. Database Management:
   - List all databases
   - Get database statistics
   - Drop (delete) a database

2. Collection Management:
   - List all collections in a database
   - Create a new collection
   - Drop (delete) a collection
   - Rename a collection
   - Get collection statistics (size, storage, etc.)
   - Get collection schema
   - Manage indexes (list, create)

3. Document Operations:
   - Find documents (with filters, projections, sorting, and limits)
   - Insert multiple documents
   - Update multiple documents
   - Delete multiple documents
   - Count documents (with optional filters)
   - Aggregate (run aggregation pipelines)

4. Logs and Monitoring:
   - Get recent MongoDB logs (global or startup warnings)

5. Search Tools:
   - Text search for solutions to alert conditions
   - Vector search for solutions to alert conditions
   - Fusion search for solutions to alert conditions

6. Utility:
   - Get the current UTC time
```

# sample prompt

```
 Please access the realtime_network_logs table in the network monitoring database and, after identifying the correct UTC timestamp field by examining its schema, retrieve event descriptions inserted within the last 20 seconds; then, summarize these descriptions, pinpointing any alert conditions where the severity is 4 or greater, and finally, leverage vector search tools to find and present possible solutions for all identified alerts.identify the towers impacted
```

# sample output

```
Query:  Please access the realtime_network_logs table in the network monitoring database and, after identifying the correct UTC timestamp field by examining its schema, retrieve event descriptions inserted within the last 20 seconds; then, summarize these descriptions, pinpointing any alert conditions where the severity is 4 or greater, and finally, leverage vector search tools to find and present possible solutions for all identified alerts.identify the towers impacted
getting embeddings :

[Calling tool list-databases with args {}]
[Calling tool list-collections with args {'database': 'network_monitoring'}]
[Calling tool collection-schema with args {'database': 'network_monitoring', 'collection': 'realtime_network_logs'}]
[Calling tool get_utc_time with args {}]
[Calling tool find with args {'database': 'network_monitoring', 'collection': 'realtime_network_logs', 'filter': {'event_timestamp': {'$gte': {'$date': '2025-07-28T19:16:10.414497+00:00'}}}, 'projection': {'event_description': 1, 'severity': 1, 'source_tower_id': 1, '_id': 0}}]
[Calling tool vector_search_tool with args {'user_input': 'Critical RF Module failure detected, impacting signal transmission and reception across multiple sectors, requiring immediate attention to restore full service capability.'}]
### Summary of Event Descriptions

**Alert Condition: Critical RF Module Failure**
- **Description**: Critical RF Module failure detected, impacting signal transmission and reception across multiple sectors, requiring immediate attention to restore full service capability.
- **Severity**: 4 and 5
- **Source Tower ID**: tower_4

### Possible Solutions for Identified Alerts

1. **RF Module Failure**:
   - **Description**: An active alarm indicating a malfunction or failure within one or more RF modules, which are often integrated into or connected to the Remote Radio Unit (RRU). This can lead to service degradation or complete outage for affected sectors.
   - **Probable Causes**:
     - Internal hardware fault within the RF module.
     - Missing or faulty connections to the module, including power, fiber, or RF jumpers.
     - Insufficient or unstable power supply to the module.
   - **Solutions**:
     - Conduct a thorough visual inspection of the RF module and all its connections, including power cables, fiber optic cables, and RF jumpers. Look for any signs of physical damage, loose connections, or improper seating.
     - If the system supports it, attempt a unit block/unblock or a soft reset of the affected RF module.
     - Ensure stable and sufficient power supply to the RF module.

### Towers Impacted
- **Tower ID**: tower_4

These solutions are derived from the manual and are intended to address the critical RF module failure impacting tower_4.
```

# chat history

chat history per session stored here

```
            database_name="historical_db",
            collection_name="chat_messages"
```
