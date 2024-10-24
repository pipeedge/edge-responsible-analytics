from prometheus_client import start_http_server, Summary, Counter, Gauge
import time
import random

# Create Prometheus metrics
REQUEST_SUMMARY = Summary('request_processing_seconds', 'Time spent processing request')
REQUEST_COUNT = Counter('requests_total', 'Total number of requests')
MODEL_UPTIME = Gauge('model_uptime_seconds', 'Uptime of the aggregated model')

@REQUEST_SUMMARY.time()
def process_request():
    """A dummy function that takes some time."""
    time.sleep(random.uniform(0.1, 0.5))

def monitor_model():
    start_time = time.time()
    while True:
        REQUEST_COUNT.inc()
        process_request()
        MODEL_UPTIME.set(time.time() - start_time)
        time.sleep(1)

if __name__ == "__main__":
    # Start up the server to expose the metrics.
    start_http_server(8000)
    # Generate some requests.
    monitor_model()

