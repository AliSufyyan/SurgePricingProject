"""
Surge Output Publisher — Kafka Producer
Reads live surge predictions from the streaming pipeline
and publishes final surge multipliers to 'surge-output' topic.
This simulates how the pricing engine would feed downstream apps
(driver app, passenger app, billing service).
 
Run: python kafka/surge_output_consumer.py
"""
 
import json
import time
import random
import datetime
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import NoBrokersAvailable
 
 
def create_clients(broker: str):
    for attempt in range(15):
        try:
            consumer = KafkaConsumer(
                "ride-requests",
                bootstrap_servers=broker,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                auto_offset_reset="latest",
                group_id="surge-output-group",
                consumer_timeout_ms=5000,
            )
            producer = KafkaProducer(
                bootstrap_servers=broker,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8"),
            )
            print(f"  Connected to Kafka at {broker}")
            return consumer, producer
        except NoBrokersAvailable:
            print(f"  Waiting for Kafka, attempt {attempt + 1}/15...")
            time.sleep(4)
    raise RuntimeError("Could not connect to Kafka.")
 
 
def compute_surge(event: dict) -> float:
    """Lightweight surge computation mirroring the Spark streaming logic."""
    requests = event.get("ride_requests", 50)
    drivers  = max(1, event.get("available_drivers", 20))
    ratio    = requests / drivers
 
    if ratio >= 5.0:   base = 3.5
    elif ratio >= 3.5: base = 2.8
    elif ratio >= 2.5: base = 2.2
    elif ratio >= 1.8: base = 1.8
    elif ratio >= 1.3: base = 1.4
    elif ratio >= 1.0: base = 1.2
    else:              base = 1.0
 
    weather_code = event.get("weather_code", 0)
    weather_premium = 0.5 if weather_code >= 3 else 0.3 if weather_code >= 2 else 0.0
 
    event_premium   = 0.5 if event.get("special_event", False) else 0.0
    weekend_premium = 0.3 if event.get("is_weekend", False)    else 0.0
 
    return round(min(5.0, base + weather_premium + event_premium + weekend_premium), 2)
 
 
def main():
    broker = "localhost:9092"
    consumer, producer = create_clients(broker)
 
    print("📡  Listening on 'ride-requests' → computing → publishing to 'surge-output'\n")
    published = 0
 
    try:
        for message in consumer:
            event = message.value
            zone  = event.get("zone", "Unknown")
 
            surge = compute_surge(event)
 
            output = {
                "timestamp":          datetime.datetime.now().isoformat(),
                "zone":               zone,
                "surge_multiplier":   surge,
                "ride_requests":      event.get("ride_requests"),
                "available_drivers":  event.get("available_drivers"),
                "weather":            event.get("weather", "clear"),
                "special_event":      event.get("special_event", False),
                "base_fare_usd":      2.50,
                "effective_fare_usd": round(2.50 * surge, 2),
                "source_event_id":    event.get("event_id", ""),
            }
 
            producer.send("surge-output", key=zone, value=output)
            published += 1
 
            bar = "█" * int((surge - 1.0) * 10)
            print(
                f"[{published:5d}] {zone:<10} | "
                f"surge={surge:.2f}× {bar:<20} | "
                f"fare=${output['effective_fare_usd']:.2f}"
            )
 
    except KeyboardInterrupt:
        print(f"\n  Stopped. Published {published} surge events.")
    finally:
        producer.flush()
        producer.close()
        consumer.close()
 
 
if __name__ == "__main__":
    main()