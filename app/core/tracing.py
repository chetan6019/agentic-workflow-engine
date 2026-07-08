"""OpenTelemetry setup — call setup_tracing(app) once at startup.

One emission path for all traces: app code opens spans with
``trace.get_tracer(__name__)`` and they are exported over OTLP to Grafana
Alloy (the collector), which forwards traces to Langfuse and metrics to
Prometheus. When OTEL_EXPORTER_OTLP_ENDPOINT is unset nothing is configured,
so every span is a cheap no-op — tests and bare dev need no collector.
"""

from __future__ import annotations

from fastapi import FastAPI
from opentelemetry import trace

from app.config import get_settings


def setup_tracing(app: FastAPI) -> None:
    """Configure the global tracer provider and instrument FastAPI.

    A no-op when the OTLP endpoint setting is unset. The SDK imports live
    inside the function so simply importing this module stays dependency-light
    for tools/tests that never enable tracing.
    """
    settings = get_settings()
    endpoint = getattr(settings, "otel_exporter_otlp_endpoint", None)
    if not endpoint:
        return

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({"service.name": settings.otel_service_name})
    provider = TracerProvider(resource=resource)
    # BatchSpanProcessor exports off the request path (background thread).
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces"))
    )
    trace.set_tracer_provider(provider)
    # Automatic server spans for every HTTP request (route, method, status).
    FastAPIInstrumentor.instrument_app(app)
