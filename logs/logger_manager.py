# logs/logger_manager.py

import os
import json
import csv
from datetime import datetime
from threading import Lock

LOG_DIR = "logs/"
os.makedirs(LOG_DIR, exist_ok=True)

log_locks = {
    "trade": Lock(),
    "performance": Lock(),
    "security": Lock(),
}

def log_trade(trade_data: dict):
    """Logs a new trade execution into trade_log.csv"""
    with log_locks["trade"]:
        file_path = os.path.join(LOG_DIR, "trade_log.csv")
        write_header = not os.path.exists(file_path)
        with open(file_path, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=trade_data.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(trade_data)

def log_performance(perf_data: dict):
    """Appends or updates performance intelligence in performance_report.json"""
    with log_locks["performance"]:
        file_path = os.path.join(LOG_DIR, "performance_report.json")
        data = {}
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}
        data[datetime.utcnow().isoformat()] = perf_data
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

def log_security_event(event_data: dict):
    """Logs security alerts, breaches, and anomalies"""
    with log_locks["security"]:
        file_path = os.path.join(LOG_DIR, "security_audit_log.json")
        logs = []
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                try:
                    logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []
        event_data["timestamp"] = datetime.utcnow().isoformat()
        logs.append(event_data)
        with open(file_path, "w") as f:
            json.dump(logs, f, indent=4)

def load_security_logs():
    """Loads all security audit logs as a list."""
    file_path = os.path.join(LOG_DIR, "security_audit_log.json")
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []
