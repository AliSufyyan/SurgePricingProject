from kafka import KafkaProducer
import json, time, random, datetime

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

Zones = ['Downtown', 'Uptown', 'Suburbs', 'Industrial']

while True:
    message = {
        "timestamp": datetime.datetime.now().isoformat(),
        "zone": random.choice(Zones),
        "ride_requests": random.randint(0, 200),
        "available_drivers": random.randint(5, 80),
        "hour": datetime.datetime.now().hour,
        "is_weekend": datetime.datetime.now().weekday() >= 5,
        "weather_code": random.choice([0, 1, 2]),
    }
    producer.send('ride_requests', value=message)
    print(f"Sent: {message}")
    time.sleep(1)   