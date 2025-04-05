from kafka import KafkaProducer
from json import dumps
import time

def init_producer(bootstrap_servers=["192.168.0.104:9092"]):
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: dumps(v).encode('utf-8')
    )

def log_request(producer, query, latency, topic="movielogN"):
    message = {
        "timestamp": time.time(),
        "endpoint": "recommendation request",
        "status": "200",
        "query": query,
        "latency_ms": latency
    }
    producer.send(topic, value=message)