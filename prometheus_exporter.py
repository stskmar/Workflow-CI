from prometheus_client import start_http_server, Gauge
import psutil
import time

cpu = Gauge("system_cpu_usage", "CPU usage")
memory = Gauge("system_memory_usage", "Memory usage")

if __name__ == "__main__":
    start_http_server(9100)
    while True:
        cpu.set(psutil.cpu_percent())
        memory.set(psutil.virtual_memory().percent)
        time.sleep(5)
