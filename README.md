## Monitoring Metrics

To ensure system health, the following metrics are monitored:

- **Model Metric**: **CPU Time (`process_cpu_seconds_total`)** — Measures total CPU usage over time, helping identify performance bottlenecks during model inference or processing.

- **Data Metric**: **Request Volume (`http_requests_total`)** — Indicates how much data (in the form of API calls) is being received, helping identify usage spikes or frontend issues.

- **Server Metric**: **Memory Usage (`process_resident_memory_bytes`)** — Tracks actual RAM usage of the application to detect potential memory leaks or excessive memory consumption.
