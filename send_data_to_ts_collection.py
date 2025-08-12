import threading
import time
import random
from datetime import datetime, timedelta
from pymongo import MongoClient
from pymongo.errors import CollectionInvalid, OperationFailure

# --- Configuration ---



NUM_TOWERS = 10  # Number of towers sending data, each will get its own thread
DOCUMENTS_PER_SEND = 50  # Number of documents each tower sends per interval
SEND_INTERVAL_SECONDS = 5
ERRORS_PER_BATCH =  0  # Number of error documents to inject per batch (must be < DOCUMENTS_PER_SEND)

import os
from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file first
MONGO_URI = os.getenv("ATLAS_URI")
DB_NAME = os.getenv("ATLAS_DB_NAME")
COLLECTION_NAME = os.getenv("ATLAS_COLLECTION_NAME")
# Ensure ERRORS_PER_BATCH is less than DOCUMENTS_PER_SEND
if ERRORS_PER_BATCH >= DOCUMENTS_PER_SEND:
    raise ValueError("ERRORS_PER_BATCH must be less than DOCUMENTS_PER_SEND")

# New error messages for severity 4 and 5, adjusted to be between 10 and 50 words
HIGH_SEVERITY_ERROR_MESSAGES = [
    "Critical RF Module failure detected, impacting signal transmission and reception across multiple sectors, requiring immediate attention to restore full service capability.",
    "Antenna VSWR reading is significantly over threshold, indicating a major impedance mismatch that could lead to power loss and potential damage to radio equipment."    ,
    "Undesirable Passive Intermodulation (PIM) has been detected, causing interference and degrading signal quality within the cell coverage area, affecting user experience."
    ,
     "The antenna tilt alarm has activated, suggesting a physical misalignment of the antenna array which is severely impacting network coverage and subscriber connectivity."
     ,
    # "RET Motor failure confirmed on an antenna, preventing remote electrical tilt adjustments and causing suboptimal signal coverage until manual repair is performed.",
    # "An antenna element failure has occurred, leading to reduced gain and directional control, severely impairing the cell's ability to provide robust wireless service.",
    # "Significant feeder cable damage detected, resulting in substantial signal attenuation and power leakage, which necessitates urgent replacement to maintain network integrity.",
    # "Weak 5G signal observed with critically low RSRP and SINR values, indicating poor reception conditions that are severely impacting high-speed data services for users.",
    # "Abnormally high electromagnetic field levels detected around the tower, possibly exceeding safety guidelines and warranting immediate investigation and mitigation actions.",
    # "Complete mains power failure at the tower site, forcing a switch to backup systems; sustained operation depends on the generator or battery capacity remaining.",
    # "Emergency Power System is now active due to grid power loss, indicating the tower is running on backup batteries or a generator, with limited operational duration remaining.",
    # "The backup generator failed to start automatically during a power outage, leaving the tower solely reliant on battery power and risking full service interruption soon.",
    # "Generator run time has exceeded its maximum continuous operational limit, suggesting a prolonged power outage or a malfunction in the primary power restoration system.",
    # "Generator fuel level is critically low, prompting an urgent refueling requirement to ensure continued operation of the tower during extended mains power interruptions.",
    # "Critically low generator oil pressure detected, which poses a serious risk of engine damage if not addressed immediately, potentially leading to complete generator shutdown.",
    # "Generator overheating alarm activated, indicating a cooling system malfunction or excessive load, threatening the generator's integrity and continuous power supply.",
    # "Generator alternator charge failure detected, meaning the generator is not recharging its own starting battery, risking future startup issues and unreliable backup power.",
    # "Rectifier system abnormal or fault condition reported, indicating issues with converting AC to DC power essential for tower equipment operation and battery charging.",
    # "Rectifier overcurrent detected, possibly due to a short circuit or an overloaded power draw, which could damage the power supply unit and other connected components.",
    # "Rectifier output voltage is abnormal, supplying incorrect power levels to sensitive tower equipment, risking hardware damage and unstable network operation.",
    # "Battery bank low voltage alarm triggered, indicating that the backup batteries are significantly depleted and may not provide sufficient power during an outage.",
    # "Excessive battery discharge alarm, suggesting a rapid drain on the backup power system, possibly due to high load or a short circuit, reducing autonomy.",
    # "Battery overcharge alarm activated, indicating a potential regulator malfunction that could damage the batteries and reduce their lifespan, posing a fire risk.",
    # "Battery overheating alarm detected, which is a critical condition that can lead to thermal runaway and permanent battery damage or even a fire hazard.",
    # "Power Distribution Unit (PDU) alarm, indicating a fault within the power distribution system, potentially affecting power delivery to various tower components.",
    # "A significant power surge has been detected, potentially damaging sensitive electronic equipment and disrupting operations; a thorough system check is recommended.",
    # "Lightning Protection System alarm has activated, suggesting a strike or a fault in the grounding system, requiring inspection to ensure continued protection.",
    # "Grounding fault detected, indicating an improper electrical connection to the earth, which can compromise equipment safety and performance, requiring urgent repair.",
    # "Baseband Unit (BBU) internal fault, indicating a critical hardware or software error within the primary processing unit, severely impacting network functionality.",
    # "Remote Radio Unit (RRU) internal fault, affecting signal processing at the antenna level, leading to degraded coverage and reduced capacity in that sector.",
    # "General hardware error detected within the BBU or RRU, indicating a component malfunction that is compromising the overall stability and performance of the system.",
    # "SFP Module fault reported, indicating an issue with the small form-factor pluggable transceiver, which is crucial for optical fiber data transmission links.",
    # "Optical Fiber Link Degradation detected, resulting in increased bit error rates and reduced throughput, impacting high-speed data backhaul and core network connectivity.",
    # "Fiber Optic Connector dirty or damaged, causing significant signal loss and connectivity issues between network elements, requiring cleaning or replacement for restoration.",
    # "CPRI/eCPRI Link Failure, indicating a critical communication breakdown between the Baseband Unit and Remote Radio Unit, leading to a complete sector outage.",
    # "Core Network Connectivity Loss, meaning the tower has lost its connection to the central network infrastructure, causing a complete service outage for all connected users.",
    # "Backhaul Link Down (Fiber/Microwave), indicating a complete loss of communication between the tower and the core network, resulting in a total service outage.",
    # "Backhaul Link Saturation due to severe congestion, leading to significantly reduced data throughput and high latency for all users connected to this tower.",
    # "Microwave Link Alignment Alarm, suggesting a physical shift in the microwave dish orientation, causing signal degradation and potential loss of backhaul capacity.",
    # "Backhaul Routing Protocol Error detected, preventing proper data packet forwarding and leading to intermittent or complete loss of connectivity for the tower.",
    # "X2 Interface Link Failure, indicating a breakdown in communication between neighboring eNBs, which impacts handover success rates and overall network mobility.",
    # "Network Element Communication Loss, signifying a critical failure in the ability of different network components to exchange data and control signals effectively.",
    # "Significantly low throughput detected, indicating network congestion or a bottleneck that is severely impacting data speeds and user experience within the cell.",
    # "Excessive packet loss detected on the network interface, leading to degraded voice quality, slow data speeds, and unreliable service for connected devices.",
    # "High latency detected in network traffic, causing noticeable delays in communication and impacting real-time applications such as voice calls and online gaming.",
    # "Elevated call drop rate observed, indicating frequent disconnections during ongoing calls, a critical issue for service quality and subscriber satisfaction.",
    # "High RRC Setup Failure rate, preventing new connection attempts from successfully establishing, which severely limits network access for new users.",
    # "General service degradation reported across the cell, affecting various network functions and user experiences, requiring broad system diagnostics.",
    # "Resource congestion detected on the tower, indicating an overload of capacity, which is leading to performance degradation for connected users.",
    # "CPU utilization on the main processing unit is critically high, threatening system stability and responsiveness, potentially leading to performance issues or crashes.",
    # "Memory utilization on network equipment is critically high, indicating a potential memory leak or insufficient resources, which can lead to system instability.",
    # "Cooling fan failure detected within the BBU or RRU, risking equipment overheating and potential hardware damage if not addressed quickly.",
    # "Cooling fan speed is critically low, indicating a fan malfunction or obstruction, which can lead to insufficient cooling and eventual equipment overheating.",
    # "Equipment overheating alarm activated, indicating temperatures exceeding safe operating limits within the hardware cabinet, posing a risk of system failure.",
    # "High temperature alarm inside the shelter or cabinet, signaling inadequate cooling or excessive heat generation, which could damage sensitive electronics.",
    # "Low temperature alarm inside the shelter or cabinet, indicating conditions that could affect equipment performance or cause component degradation over time.",
    # "Humidity alarm within the shelter or cabinet, warning of condensation or moisture levels that could lead to electrical shorts and corrosion of components.",
    # "Flood sensor alarm triggered in the shelter or cabinet, indicating water ingress that poses an immediate threat of severe equipment damage and power hazards.",
    # "Physical intrusion detection alarm at the tower site, indicating unauthorized access or tampering with equipment, requiring immediate security response.",
    # "Tower door open alarm, signaling an unsecured entry point which could allow unauthorized access to sensitive equipment and compromise site security.",
    # "Vibration sensor alarm, indicating unusual movement or instability of the tower structure, possibly due to external factors or structural integrity issues.",
    # "Smoke or heat detector alarm activated inside the tower facility, indicating a potential fire hazard and requiring immediate emergency response and investigation.",
    # "Fire Suppression System alarm or discharge, confirming activation of the fire extinguishing system, indicating a significant fire event occurred at the site.",
    # "5G Network Parameter Mismatch detected between network elements, potentially causing handover failures, suboptimal routing, and service disruptions.",
    # "Network Slicing Configuration Error, preventing proper isolation and resource allocation for specific 5G services, impacting quality of service guarantees.",
    # "gNB Software Upgrade Failure, indicating a failed attempt to update the next-generation NodeB software, which could leave the system in an unstable state.",
    # "Remote Reset Failure, where an attempt to remotely reboot or reset a network element was unsuccessful, requiring a physical presence to resolve the issue.",
    # "Configuration Backup Failure, meaning recent system configuration changes were not successfully saved, risking data loss during a system restore or catastrophic event.",
    # "License Expiry Warning for a critical software feature, indicating that functionality may cease soon if the license is not renewed, impacting network services.",
    # "Security Certificate Invalid or Expired, compromising secure communication channels and potentially exposing the network to unauthorized access or data breaches.",
    # "Denial of Service (DoS) Attack Detection, indicating a malicious attempt to overwhelm the tower's resources and disrupt network services, requiring immediate mitigation.",
    # "GPS Signal Loss Alarm, indicating the loss of satellite synchronization, which can severely impact network timing and the accuracy of location-based services.",
    # "GPS Antenna Fault, preventing the tower from receiving accurate timing signals from satellites, potentially affecting network synchronization and performance.",
    # "PTP Synchronization Alarm, signaling a disruption in the Precision Time Protocol synchronization, which is critical for accurate timing in 5G networks.",
    # "NTP Synchronization Failure, indicating that the tower cannot properly synchronize its time with Network Time Protocol servers, affecting log accuracy and data integrity."
]

# Event to signal threads to shut down gracefully
shutdown_event = threading.Event()

# --- MongoDB Setup ---

def get_mongo_collection():
    """
    Connects to MongoDB and returns the time series collection.
    If the collection doesn't exist, it attempts to create it as a time series collection.
    """
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]

        if COLLECTION_NAME not in db.list_collection_names():
            print(f"Collection '{COLLECTION_NAME}' does not exist. Attempting to create as time series collection...")
            try:
                db.create_collection(
                    COLLECTION_NAME,
                    timeseries={
                        "timeField": "event_timestamp",
                        "granularity": "seconds"
                    },
                    expireAfterSeconds=60 * 60 * 24 * 30  # Data expires after 30 days (optional)
                )
                print(f"Time series collection '{COLLECTION_NAME}' created successfully.")
            except CollectionInvalid as e:
                print(f"Error creating collection: {e}. It might already exist but not as time series, or there's a configuration issue.")
            except Exception as e:
                print(f"An unexpected error occurred during collection creation: {e}")
        else:
            print(f"Collection '{COLLECTION_NAME}' already exists.")

        return db[COLLECTION_NAME], client
    except Exception as e:
        print(f"Failed to connect to MongoDB or get collection: {e}")
        exit(1)

# --- Data Generation Function ---

def generate_tower_data(tower_id, is_error=False):
    """
    Generates a single data document for a given tower.
    Timestamps are always in UTC.
    """
    timestamp_dt = datetime.utcnow()
    severity = random.randint(0, 3)
    category = random.choice(["STAT", "SymmetricDS"])

    event_description = "Normal operation data"
    if is_error:
        event_description = random.choice(HIGH_SEVERITY_ERROR_MESSAGES)
        severity = random.randint(4, 5)
    else:
        possible_descriptions = [
            "Routine system check completed successfully, confirming all network parameters are well within expected operational ranges, ensuring stable and reliable service for connected users.",
            "Normal data traffic patterns observed across all active channels, indicating healthy network performance and optimal user connectivity throughout the designated coverage area.",
            "Regular maintenance log update processed, confirming no immediate issues or unexpected anomalies were detected during the recent automated diagnostic scans and system health checks.",
            "Baseline performance metrics recorded, consistently showing robust network throughput and minimal latency, which is highly ideal for current operational demands and user experience.",
            "Connectivity tests passed without any errors or interruptions, affirming robust backhaul links and seamless integration with the core network infrastructure, ensuring continuous service.",
            "Automated system health report generated, indicating a clear 'green' status for all critical components and essential subsystems within the telecom tower unit, confirming full functionality.",
            "Environmental sensors report stable conditions, with both temperature and humidity levels comfortably well within the specified operational thresholds for continued optimal function of all equipment.",
            "Power consumption levels are perfectly stable and remain well within acceptable limits, reflecting highly efficient energy management and a consistently reliable power supply to all tower equipment.",
            "Security protocols continuously monitored and rigorously enforced; absolutely no unusual activity, suspicious patterns, or unauthorized access attempts have been detected in the last reporting cycle.",
            "Software version integrity verified across the entire system, confirming that all running applications and firmware are fully up-to-date and operating without any known vulnerabilities or unexpected glitches."
        ]
        event_description = random.choice(possible_descriptions)

    document = {
        "source_tower_id": f"tower_{tower_id}",
        "event_id": str(random.randint(10000, 99999)),
        "event_description": event_description,
        "category": category,
        "severity": severity,
        "event_timestamp": timestamp_dt
    }
    return document

# --- MongoDB Insertion Logic ---

def produce_tower_data(tower_id, mongo_collection, shutdown_event, errors_per_batch):
    """
    Function for each tower thread to generate and insert data into MongoDB.
    Injects a fixed number of error documents per batch.
    """
    while not shutdown_event.is_set():
        documents_to_send = []
        
        # Choose random indices for error injection
        error_indices = random.sample(range(DOCUMENTS_PER_SEND), min(errors_per_batch, DOCUMENTS_PER_SEND))
        
        if errors_per_batch > 0:
            print(f"--- Tower {tower_id}: Injecting {len(error_indices)} error(s) into batch ---")
        
        error_timestamps = []
        for i in range(DOCUMENTS_PER_SEND):
            is_error_doc = (i in error_indices)
            doc = generate_tower_data(tower_id, is_error=is_error_doc)
            documents_to_send.append(doc)
            '''
            if is_error_doc:
                error_timestamps.append(doc['fieldtime'])
                '''
        
        try:
            if documents_to_send:
                mongo_collection.insert_many(documents_to_send)
                
                error_doc_info = ""
                # if error_timestamps:
                #    error_doc_info = f", Injected error timestamps: {', '.join(str(ts) for ts in error_timestamps)}"
                                
                print(f"Tower {tower_id}: Inserted {len(documents_to_send)} documents. {error_doc_info}")
        except OperationFailure as e:
            print(f"Tower {tower_id}: MongoDB insert failed: {e}")
        except Exception as e:
            print(f"Tower {tower_id}: Unexpected error during MongoDB insert: {e}")

        shutdown_event.wait(SEND_INTERVAL_SECONDS)

    print(f"Tower {tower_id}: Shutting down data generation thread.")

def main():
    mongo_collection, mongo_client = get_mongo_collection()
    if mongo_collection is None:
        print("Failed to get MongoDB collection. Exiting.")
        return

    threads = []
    for i in range(1, NUM_TOWERS + 1):
        thread = threading.Thread(
            target=produce_tower_data,
            args=(i, mongo_collection, shutdown_event, ERRORS_PER_BATCH)
        )
        threads.append(thread)
        thread.start()

    # Set the program to run for 30 mins (1800 seconds)
    # so that the program doesnot accidnetally 
    run_duration_seconds = 1800
    start_time = time.time()

    try:
        while time.time() - start_time < run_duration_seconds:
            if not any(thread.is_alive() for thread in threads):
                break # All threads finished before the timer
            time.sleep(1)
        
        # If the loop finishes due to the timer, signal shutdown
        if time.time() - start_time >= run_duration_seconds:
            print("\n1 hour has passed. Signaling threads to shut down...")
            shutdown_event.set()

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Signaling threads to shut down...")
        shutdown_event.set()
    finally:
        # Wait for all threads to complete or timeout
        for thread in threads:
            if thread.is_alive():
                print(f"Waiting for thread {thread.name} to finish...")
                thread.join(timeout=SEND_INTERVAL_SECONDS + 5)
                if thread.is_alive():
                    print(f"Warning: Thread {thread.name} did not shut down gracefully. It might be stuck.")
        print("All producer threads have terminated or timed out.")
        if mongo_client:
            mongo_client.close()
            print("MongoDB client closed.")
        print("Application exiting.")

if __name__ == "__main__":
    main()