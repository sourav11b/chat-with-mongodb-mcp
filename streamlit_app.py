import streamlit as st
import asyncio
import nest_asyncio
import traceback
import os
import time
import threading
import random
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pymongo import MongoClient, DESCENDING as pymongo_DESCENDING
from pymongo.errors import CollectionInvalid, OperationFailure
from dotenv import load_dotenv
import json

# Apply nest_asyncio immediately to allow nested event loops in Streamlit
nest_asyncio.apply()

# --- 1. STREAMLIT PAGE CONFIGURATION (MUST BE FIRST) ---
st.set_page_config(
    page_title="MongoDB AI Assistant & Data Controller",
    layout="wide"
)

# --- 2. CONFIGURATION and MONGODB CLIENTS ---

# Load environment variables (MONGO_URI, DB_NAME, COLLECTION_NAME)
load_dotenv() 

ATLAS_URI = os.getenv("ATLAS_URI")
ATLAS_DB_NAME = os.getenv("ATLAS_DB_NAME")
# Primary Collection for the Chatbot Tool
ATLAS_COLLECTION_NAME_CHATBOT = os.getenv("ATLAS_COLLECTION_NAME") 
# Realtime Collection for the Dashboard/Producer
ATLAS_COLLECTION_NAME_REALTIME = os.getenv("ATLAS_COLLECTION_NAME_REALTIME", "realtime_network_logs")


# Mongo Client Initialization (Shared)
@st.cache_resource
def get_mongo_client():
    """Connects to MongoDB and returns the client."""
    if not ATLAS_URI:
        st.error("ATLAS_URI environment variable is not set.")
        return None
    try:
        # NOTE: Added connect=False to prevent connection issues in some environments
        client = MongoClient(ATLAS_URI, connect=False)
        return client
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {e}")
        return None

mongo_client = get_mongo_client()

if not mongo_client:
    st.stop()
    
# Collection for the Dashboard and Producer
realtime_network_logs_collection = mongo_client[ATLAS_DB_NAME][ATLAS_COLLECTION_NAME_REALTIME]


# 2. Producer Configuration
NUM_TOWERS = 10 
DOCUMENTS_PER_SEND = 50 
SEND_INTERVAL_SECONDS = 5
ERRORS_PER_BATCH_NORMAL = 0
ERRORS_PER_BATCH_INJECT = 9

HIGH_SEVERITY_ERROR_MESSAGES = [
    "Critical RF Module failure detected, impacting signal transmission and reception across multiple sectors, requiring immediate attention to restore full service capability.",
    "Antenna VSWR reading is significantly over threshold, indicating a major impedance mismatch that could lead to power loss and potential damage to radio equipment.",
    "Undesirable Passive Intermodulation (PIM) has been detected, causing interference and degrading signal quality within the cell coverage area, affecting user experience.",
    "The antenna tilt alarm has activated, suggesting a physical misalignment of the antenna array which is severely impacting network coverage and subscriber connectivity.",
    "RET Motor failure confirmed on an antenna, preventing remote electrical tilt adjustments and causing suboptimal signal coverage until manual repair is performed."
]

# 3. Dashboard Configuration
LOG_WINDOW_MINUTES = 0.5


# --- 1. PRODUCER LOGIC ---

def initialize_collection(db_name, collection_name, client):
    """
    Connects to MongoDB and attempts to create the collection as a time series.
    """
    try:
        db = client[db_name]
        if collection_name not in db.list_collection_names():
            try:
                db.create_collection(
                    collection_name,
                    timeseries={
                        "timeField": "event_timestamp",
                        "granularity": "seconds"
                    },
                    expireAfterSeconds=60 * 60 * 24 * 30 
                )
                print(f"Time series collection '{collection_name}' created successfully.")
            except CollectionInvalid as e:
                print(f"Error creating collection: {e}.")
    except Exception as e:
        print(f"Failed to initialize collection: {e}")

# Call once to ensure the collection exists before producer starts
initialize_collection(ATLAS_DB_NAME, ATLAS_COLLECTION_NAME_REALTIME, mongo_client)

def generate_tower_data(tower_id: int, is_error: bool) -> Dict[str, Any]:
    """Generates a single data document for a given tower."""
    timestamp_dt = datetime.utcnow()
    
    if is_error:
        event_description = random.choice(HIGH_SEVERITY_ERROR_MESSAGES)
        severity = random.randint(4, 5) 
    else:
        possible_descriptions = [
            "Routine system check completed successfully...",
            "Normal data traffic patterns observed...",
            "Regular maintenance log update processed...",
            "Baseline performance metrics recorded...",
            "Automated system health report generated, indicating a clear 'green' status...",
        ]
        event_description = random.choice(possible_descriptions)
        severity = random.randint(0, 3)

    document = {
        "source_tower_id": f"tower_{tower_id}",
        "event_id": str(random.randint(10000, 99999)),
        "event_description": event_description,
        "category": random.choice(["STAT", "SymmetricDS"]),
        "severity": severity,
        "event_timestamp": timestamp_dt
    }
    return document

def produce_data_worker(
    tower_id: int, 
    mongo_collection: Any, 
    shutdown_event: threading.Event, 
    errors_per_batch: int
):
    """Thread worker function to generate and insert data."""
    while not shutdown_event.is_set():
        documents_to_send = []
        
        error_indices = random.sample(range(DOCUMENTS_PER_SEND), min(errors_per_batch, DOCUMENTS_PER_SEND))
        
        # In error injection mode, focus errors on a small subset of towers (e.g., 2, 7)
        current_tower_id = random.choice([2, 7]) if errors_per_batch == ERRORS_PER_BATCH_INJECT else tower_id
        
        for i in range(DOCUMENTS_PER_SEND):
            is_error_doc = (i in error_indices)
            doc = generate_tower_data(current_tower_id, is_error=is_error_doc)
            documents_to_send.append(doc)
        
        try:
            if documents_to_send:
                mongo_collection.insert_many(documents_to_send)
        except OperationFailure as e:
            print(f"Tower {current_tower_id}: MongoDB insert failed: {e}")
        except Exception as e:
            print(f"Tower {current_tower_id}: Unexpected error during MongoDB insert: {e}")

        shutdown_event.wait(SEND_INTERVAL_SECONDS)


# --- 2. DASHBOARD / CHAT LOGIC ---

try:
    from run_chat import chat_loop 
except ImportError:
    st.error("Warning: Could not import **'chat_loop'** from 'run_chat.py'. LLM Chat Assistant will be disabled.")
    chat_loop = None 
except Exception as e:
    st.error(f"Error during import from run_chat.py: {e}")
    st.stop()


def get_raw_critical_logs():
    """Fetches the raw, individual log records that are critical (severity >= 4)."""
    try:
        time_threshold = datetime.utcnow() - timedelta(minutes=LOG_WINDOW_MINUTES) 

        query_filter = {
            "event_timestamp": { "$gte": time_threshold },
            "severity": { "$gte": 4 }
        }

        sort_order = [
            ("event_timestamp", pymongo_DESCENDING),
            ("severity", pymongo_DESCENDING)
        ]

        results = list(
            realtime_network_logs_collection
            .find(query_filter, {"_id": 0})
            .sort(sort_order)
        )
        return results
    except Exception as e:
        print(f"Error during raw log retrieval: {e}")
        return []
        
def get_raw_logs():
    """Fetches the raw logs within the specified window."""
    try:
        time_threshold = datetime.utcnow() - timedelta(minutes=LOG_WINDOW_MINUTES) 

        query_filter = {
            "event_timestamp": { "$gte": time_threshold }
        }

        sort_order = [
            ("event_timestamp", pymongo_DESCENDING),
            ("severity", pymongo_DESCENDING)
        ]

        results = list(
            realtime_network_logs_collection
            .find(query_filter, {"_id": 0})
            .sort(sort_order)
        )
        return results
    except Exception as e:
        print(f"Error during raw log retrieval: {e}")
        return []

def group_events_by_tower_and_description():
    """Aggregates critical events by source tower ID and description."""
    try:
        time_threshold = datetime.utcnow() - timedelta(minutes=LOG_WINDOW_MINUTES) 

        pipeline = [
            {
                "$match": {
                    "event_timestamp": { "$gte": time_threshold },
                    "severity": { "$gte": 4 }
                }
            },
            {
                "$group": {
                    "_id": {
                        "source_tower_id": "$source_tower_id",
                        "event_description": "$event_description"
                    },
                    "count": { "$sum": 1 }
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "source_tower_id": "$_id.source_tower_id",
                    "event_description": "$_id.event_description",
                    "count": "$count"
                }
            },
            {
                "$sort": { "count": -1, "source_tower_id": 1 }
            }
        ]

        results = list(realtime_network_logs_collection.aggregate(pipeline))
        return results
    except Exception as e:
        print(f"An error occurred during aggregation: {e}")
        return []

def default_mongo_encoder(obj):
    """Custom JSON encoder to handle datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def get_latest_documents(db_name: str, collection_name: str, client: MongoClient, limit: int = 10) -> List[str]:
    """Fetches latest documents as JSON strings."""
    json_strings: List[str] = []
    try:
        collection = client[db_name][collection_name]
        try:
            raw_results = list(
                collection
                .find({}, {"_id": 0})
                .sort([("event_timestamp", pymongo_DESCENDING)]) 
                .limit(limit)
            )
        except OperationFailure:
             raw_results = list(
                collection
                .find({}, {"_id": 0})
                .sort([("_id", pymongo_DESCENDING)]) 
                .limit(limit)
            )
        
        for doc in raw_results:
            try:
                json_string = json.dumps(doc, default=default_mongo_encoder, indent=2)
                json_strings.append(json_string)
            except Exception as e:
                print(f"Error serializing document in {collection_name}: {e}")

        return json_strings
    except Exception as e:
        print(f"Error fetching documents from {collection_name}: {e}")
        return []


# --- Chat Loop Wrapper ---

def run_chat_loop_sync(query):
    """A synchronous wrapper for the async chat_loop."""
    if not chat_loop:
        return ["Error: 'chat_loop' function is unavailable."]
        
    try:
        return asyncio.run(chat_loop(query))
    except Exception as e:
        return [f"An error occurred during chat_loop: {e}\n{traceback.format_exc()}"]


# --- 3. PRODUCER CONTROLLER FUNCTIONS ---

if 'producer_threads' not in st.session_state:
    st.session_state.producer_threads: List[threading.Thread] = []
if 'shutdown_event' not in st.session_state:
    st.session_state.shutdown_event: Optional[threading.Event] = None
if 'is_running' not in st.session_state:
    st.session_state.is_running: bool = False
if 'current_mode' not in st.session_state:
    st.session_state.current_mode: str = "OFF"

def start_producer(mode: str):
    stop_producer(wait=False) 
    errors_per_batch = ERRORS_PER_BATCH_INJECT if mode == 'INJECT_ERRORS' else ERRORS_PER_BATCH_NORMAL
    st.session_state.shutdown_event = threading.Event()
    st.session_state.producer_threads = []
    
    for i in range(1, NUM_TOWERS + 1):
        thread = threading.Thread(
            target=produce_data_worker,
            args=(i, realtime_network_logs_collection, st.session_state.shutdown_event, errors_per_batch),
            daemon=True 
        )
        st.session_state.producer_threads.append(thread)
        thread.start()

    st.session_state.is_running = True
    st.session_state.current_mode = mode
    st.toast(f"Starting producer: {mode}", icon="🚀")
    st.session_state.last_dashboard_update = 0 
    st.rerun()

def stop_producer(wait=True):
    if st.session_state.producer_threads and st.session_state.shutdown_event:
        st.session_state.shutdown_event.set() 
        st.toast("Stopping producer...", icon="⏳")
        for thread in st.session_state.producer_threads:
            if thread.is_alive():
                thread.join(timeout=2) 
            
    st.session_state.producer_threads = []
    st.session_state.shutdown_event = None
    st.session_state.is_running = False
    st.session_state.current_mode = "OFF"
    if wait:
        st.rerun()


# --- 4. UI LOGIC ---

st.title("💡 MongoDB Atlas AI Agent with Real-time Monitoring and Actions")

if 'chat_response' not in st.session_state:
    st.session_state.chat_response = None
if 'input_text_value' not in st.session_state:
    st.session_state.input_text_value = "" 
if 'last_dashboard_update' not in st.session_state:
    st.session_state.last_dashboard_update = 0 
if 'raw_logs' not in st.session_state:
    st.session_state.raw_logs = []
if 'grouped_events' not in st.session_state:
    st.session_state.grouped_events = []
if 'log_unfiltered' not in st.session_state:
    st.session_state.log_unfiltered = []
if 'latest_findings' not in st.session_state:
    st.session_state.latest_findings = []
if 'latest_chat_messages' not in st.session_state:
    st.session_state.latest_chat_messages = []

st.markdown("## ⚙️ Data Producer Control")
col_p1, col_p2, col_p3 = st.columns(3)

if st.session_state.is_running:
    status_text = f"🟢 **RUNNING** (`{st.session_state.current_mode}` mode)"
    col_p1.success(f"**Status:** {status_text}")
else:
    col_p1.error(f"**Status:** 🔴 **STOPPED**")

with col_p2:
    if st.button("Start Normal Data", disabled=st.session_state.is_running, use_container_width=True, type="primary"):
        start_producer('SEND_DATA')

with col_p3:
    if st.button("Start Error Injection", disabled=st.session_state.is_running, use_container_width=True, type="primary"):
        start_producer('INJECT_ERRORS')

if st.session_state.is_running:
    if st.button("🔴 Stop Producer", use_container_width=True):
        stop_producer()

st.markdown("---") 

tab_dashboard, tab_chat, tab_raw_data = st.tabs(["📊 Real-time Dashboard", "🤖 LLM Chat Agent", "📑 Collection Logs"])

with tab_dashboard:
    col_data, _ = st.columns([2, 1]) 
    with col_data:
        st.subheader("📊 Real-time Critical Events Dashboard")
        if st.button("Refresh Dashboard 🔄"):
            st.session_state.last_dashboard_update = 0
            st.rerun() 

        if st.session_state.last_dashboard_update == 0:
            with st.spinner("Refreshing..."):
                st.session_state.raw_logs = get_raw_critical_logs()
                st.session_state.grouped_events = group_events_by_tower_and_description()
                st.session_state.log_unfiltered = get_raw_logs()
                st.session_state.last_dashboard_update = time.time()
        
        st.markdown(f"*(Last updated: **{datetime.fromtimestamp(st.session_state.last_dashboard_update).strftime('%H:%M:%S')}** UTC)*")
            
        st.markdown("##### 🗼 All Logs")
        st.dataframe(st.session_state.log_unfiltered, use_container_width=True, height=250)
            
        st.markdown("##### 🗼 Raw Critical Logs")
        st.dataframe(st.session_state.raw_logs, use_container_width=True, height=250)

        st.markdown("---")
        st.markdown("##### 🗼 Grouped Events")
        if st.session_state.grouped_events:
            df_grouped = pd.DataFrame(st.session_state.grouped_events)
            df_chart = df_grouped.pivot_table(index='source_tower_id', columns='event_description', values='count', fill_value=0)
            st.bar_chart(df_chart, use_container_width=True)
            st.dataframe(df_grouped, use_container_width=True)

with tab_chat:
    st.subheader("🤖 LLM Chatbot Agent")
    if chat_loop:
        user_query = st.text_area("Your Query:", value=st.session_state.input_text_value, height=150)
        if st.button("Submit Query", type="primary"):
            if user_query:
                # The spinner will clear once st.rerun() is hit or the block finishes
                with st.spinner("Processing..."):
                    response_array = run_chat_loop_sync(user_query)
                    st.session_state.chat_response = "\n\n" + ("\n" + "---" * 15 + "\n").join(response_array)
                
                # FORCE UI UPDATE: Clears spinner and renders results immediately
                st.rerun()
            else:
                st.warning("Please enter a query.")

        if st.session_state.chat_response:
            st.markdown("---")
            st.markdown("##### 💡 LLM Chat Response & Tool Execution Log")
            st.code(st.session_state.chat_response, language="text")

with tab_raw_data:
    st.subheader("📑 Latest MongoDB Collection Documents")
    if st.button("Fetch Latest Documents 📥"):
        st.session_state.latest_findings = get_latest_documents(ATLAS_DB_NAME, "findings", mongo_client, 10)
        st.session_state.latest_chat_messages = get_latest_documents("historical_db", "chat_messages", mongo_client, 10)
        st.rerun()
        
    st.markdown(f"##### 🔍 `{ATLAS_DB_NAME}.findings`")
    if st.session_state.latest_findings:
        st.code("[\n" + ",\n".join(st.session_state.latest_findings) + "\n]", language="json")
        
    st.markdown(f"##### 💬 `historical_db.chat_messages`")
    if st.session_state.latest_chat_messages:
        st.code("[\n" + ",\n".join(st.session_state.latest_chat_messages) + "\n]", language="json")