# Traceloop + OpenLLMetry Integration Guide

**Complete integration reference for Traceloop + OpenLLMetry with GenOps governance**

This comprehensive guide covers all aspects of integrating Traceloop and OpenLLMetry with GenOps for enterprise-grade LLM observability, cost intelligence, and governance automation.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)  
3. [Installation & Setup](#installation--setup)
4. [Basic Integration](#basic-integration)
5. [Advanced Configuration](#advanced-configuration)
6. [API Reference](#api-reference)
7. [Production Deployment](#production-deployment)
8. [Performance & Scaling](#performance--scaling)
9. [Troubleshooting](#troubleshooting)
10. [Migration & Compatibility](#migration--compatibility)

---

## Overview

### What is This Integration?

The Traceloop + OpenLLMetry integration with GenOps provides a unified approach to LLM observability with enterprise governance:

- **OpenLLMetry**: Open-source LLM observability framework (Apache 2.0)
- **Traceloop**: Commercial platform with advanced insights and analytics
- **GenOps**: Governance, cost intelligence, and policy enforcement layer

### Key Benefits

- **Enhanced Observability**: OpenLLMetry traces with governance attributes
- **Cost Intelligence**: Automatic cost attribution and budget enforcement
- **Policy Compliance**: Real-time governance and audit capabilities
- **Enterprise Readiness**: Production-grade patterns and high-availability
- **Vendor Neutral**: OpenTelemetry-native, works with all observability backends

---

## Architecture

### Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Application                         │
├─────────────────────────────────────────────────────────────┤
│  GenOps Traceloop Adapter                                   │
│  ├── Auto-Instrumentation ──┐                              │
│  ├── Manual Instrumentation ├─────────────────────┐        │
│  └── Governance Engine ──────┘                     │        │
├─────────────────────────────────────────────────────────────┤
│  OpenLLMetry Foundation                             │        │
│  ├── OpenTelemetry Instrumentation                │        │
│  ├── LLM Provider Adapters                        │        │
│  └── Trace Collection ─────────────────────────────┼────────┤
├─────────────────────────────────────────────────────────────┤
│  Traceloop Platform (Optional)                     │        │
│  ├── Advanced Analytics                            │        │
│  ├── Team Collaboration                            │        │
│  └── Model Experimentation                         │        │
├─────────────────────────────────────────────────────────────┤
│  Observability Backends                            │        │
│  ├── Datadog, Honeycomb, Grafana ─────────────────┘        │
│  ├── Custom OTLP Endpoints                                 │
│  └── Enterprise Observability Stacks                       │
└─────────────────────────────────────────────────────────────┘
```

### Component Overview

| Component | Purpose | Required |
|-----------|---------|----------|
| **GenOpsTraceloopAdapter** | Main integration adapter with governance | ✅ |
| **OpenLLMetry** | Open-source LLM observability foundation | ✅ |
| **Traceloop SDK** | Commercial platform integration | Optional |
| **OpenTelemetry** | Industry-standard observability protocol | ✅ |
| **AI Provider SDKs** | OpenAI, Anthropic, etc. | At least one |

---

## Installation & Setup

### Prerequisites

- **Python 3.8+**
- **AI Provider Account** (OpenAI, Anthropic, etc.)
- **Optional**: Traceloop Platform Account

### Installation

```bash
# Full installation with all features
pip install genops[traceloop]

# This includes:
# - OpenLLMetry (open-source framework)
# - Traceloop SDK (commercial platform integration)
# - GenOps governance enhancements
```

### Environment Configuration

```bash
# Required: AI Provider API Keys
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"  # Optional

# Optional: Traceloop Commercial Platform
export TRACELOOP_API_KEY="your-traceloop-api-key"
export TRACELOOP_BASE_URL="https://app.traceloop.com"  # Default

# Optional: GenOps Governance Defaults
export GENOPS_TEAM="your-team"
export GENOPS_PROJECT="your-project"
export GENOPS_ENVIRONMENT="production"
```

### Validation

```python
# Run comprehensive setup validation
from genops.providers.traceloop_validation import validate_setup, print_validation_result

result = validate_setup()
print_validation_result(result, detailed=True)
```

---

## Basic Integration

### Zero-Code Auto-Instrumentation (Recommended)

```python
from genops.providers.traceloop import auto_instrument

# Enable governance for ALL OpenLLMetry operations
auto_instrument(
    team="your-team",
    project="your-project",
    environment="production"
)

# Your existing OpenLLMetry code now includes governance
import openai
client = openai.OpenAI()

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello world!"}]
)
# ✅ Automatically tracked with cost attribution and governance
```

### Manual Instrumentation

```python
from genops.providers.traceloop import instrument_traceloop

# Create adapter with custom configuration
adapter = instrument_traceloop(
    team="engineering-team",
    project="llm-chatbot",
    environment="production",
    customer_id="enterprise-123",
    cost_center="r-and-d",
    
    # Governance settings
    enable_governance=True,
    daily_budget_limit=100.0,
    max_operation_cost=5.0,
    enable_cost_alerts=True
)

# Enhanced operation tracking
with adapter.track_operation(
    operation_type="chat_completion",
    operation_name="customer_query",
    tags={"priority": "high", "use_case": "support"}
) as span:
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Complex analysis request"}],
        max_tokens=200
    )
    
    # Add custom governance attributes
    span.add_attributes({
        "business.customer_tier": "enterprise",
        "business.revenue_impact": 5000.0,
        "quality.satisfaction_score": 0.95
    })
```

---

## Advanced Configuration

### Enterprise Governance Settings

```python
adapter = instrument_traceloop(
    # Core attribution
    team="platform-engineering",
    project="production-llm-api",
    environment="production",
    customer_id="multi-tenant",
    cost_center="platform-ops",
    
    # Budget and cost controls
    daily_budget_limit=500.0,        # $500 daily limit
    max_operation_cost=10.0,         # $10 per operation limit
    cost_alert_threshold=50.0,       # Alert above $50
    
    # Governance policies
    governance_policy="enforced",    # strict, advisory, audit_only
    require_cost_approval=True,      # Require approval for high-cost ops
    cost_approval_threshold=25.0,    # Approval needed above $25
    
    # Performance settings
    max_concurrent_operations=200,
    operation_timeout=300,           # 5 minutes
    retry_attempts=3,
    
    # Compliance settings
    audit_all_operations=True,
    compliance_frameworks=["SOC2", "GDPR", "HIPAA"],
    data_residency_requirements=["US", "EU"],
    
    # Traceloop platform integration
    enable_traceloop_platform=True,
    enable_advanced_analytics=True,
    enable_team_collaboration=True,
    enable_model_experimentation=True
)
```

### Multi-Provider Cost Tracking

```python
from genops.providers.traceloop import multi_provider_cost_tracking

# Enable unified cost tracking across providers
cost_summary = multi_provider_cost_tracking(
    providers=["openai", "anthropic", "gemini"],
    team="multi-provider-team",
    project="provider-comparison",
    environment="production",
    
    # Unified governance
    daily_budget_limit=200.0,
    enable_cost_alerts=True,
    governance_policy="enforced"
)

# Use different providers with unified tracking
import openai
import anthropic

openai_client = openai.OpenAI()
anthropic_client = anthropic.Anthropic()

# Both operations tracked with unified governance
openai_response = openai_client.chat.completions.create(...)
anthropic_response = anthropic_client.messages.create(...)
```

### Production High-Availability Configuration

```python
from genops.providers.traceloop import instrument_traceloop

# Production-grade configuration
adapter = instrument_traceloop(
    team="production-ops",
    project="enterprise-llm",
    environment="production",
    
    # High availability
    enable_ha=True,
    failover_regions=["us-west-2", "eu-west-1"],
    health_check_interval=30,
    
    # Performance optimization
    max_concurrent_operations=500,
    enable_batching=True,
    batch_size=100,
    batch_timeout=5000,  # 5 seconds
    
    # Circuit breaker
    circuit_breaker_threshold=10,
    circuit_breaker_timeout=60,
    
    # Monitoring and alerting
    enable_detailed_metrics=True,
    metrics_retention_days=90,
    alert_on_anomalies=True,
    
    # Security
    encrypt_sensitive_data=True,
    enable_audit_logging=True
)
```

---

## API Reference

### GenOpsTraceloopAdapter

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `team` | `str` | Required | Team name for cost attribution |
| `project` | `str` | Required | Project name for cost tracking |
| `environment` | `str` | `"development"` | Environment (dev/staging/prod) |
| `customer_id` | `Optional[str]` | `None` | Customer ID for per-customer attribution |
| `cost_center` | `Optional[str]` | `None` | Cost center for financial reporting |
| `daily_budget_limit` | `Optional[float]` | `None` | Daily spending limit in USD |
| `max_operation_cost` | `Optional[float]` | `None` | Maximum cost per operation |
| `governance_policy` | `GovernancePolicy` | `ADVISORY` | Policy enforcement level |
| `enable_cost_alerts` | `bool` | `True` | Enable cost-based alerting |
| `enable_traceloop_platform` | `bool` | `None` | Enable commercial platform features |

#### Methods

##### `track_operation(operation_type, operation_name, **kwargs)`

Track an LLM operation with governance.

```python
with adapter.track_operation(
    operation_type="chat_completion",
    operation_name="customer_query",
    tags={"priority": "high"},
    max_cost=2.0
) as span:
    # Your LLM operation here
    pass
```

**Parameters:**
- `operation_type`: Type of operation (string or TraceloopOperationType)
- `operation_name`: Name for identification
- `tags`: Additional metadata tags
- `max_cost`: Maximum allowed cost for this operation

**Returns:** Enhanced span context manager

##### `get_metrics()`

Get current governance metrics.

```python
metrics = adapter.get_metrics()
# Returns: {
#   "daily_usage": float,
#   "operation_count": int,
#   "budget_remaining": float,
#   "governance_enabled": bool,
#   ...
# }
```

### Convenience Functions

#### `auto_instrument(team, project, **kwargs)`

Enable automatic instrumentation for all OpenLLMetry operations.

```python
auto_instrument(
    team="your-team",
    project="your-project",
    environment="production",
    daily_budget_limit=100.0
)
```

#### `instrument_traceloop(**kwargs)`

Create and configure a GenOps Traceloop adapter.

```python
adapter = instrument_traceloop(
    team="your-team",
    project="your-project"
)
```

#### `multi_provider_cost_tracking(providers, **kwargs)`

Enable unified cost tracking across multiple providers.

```python
cost_summary = multi_provider_cost_tracking(
    providers=["openai", "anthropic"],
    team="multi-team",
    project="comparison"
)
```

### EnhancedSpan

Enhanced span with governance capabilities.

#### Methods

##### `update_cost(cost: float)`

Update the estimated cost for this operation.

##### `update_token_usage(input_tokens: int, output_tokens: int)`

Update token usage metrics.

##### `add_attributes(attributes: Dict[str, Any])`

Add custom attributes to the span.

##### `get_metrics() -> Dict[str, Any]`

Get comprehensive metrics for this span.

---

## Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

# Install GenOps with Traceloop
RUN pip install genops[traceloop]

# Set environment variables
ENV GENOPS_TEAM=production-team
ENV GENOPS_PROJECT=llm-service
ENV GENOPS_ENVIRONMENT=production

# Your application code
COPY . /app
WORKDIR /app

CMD ["python", "app.py"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-service
  template:
    metadata:
      labels:
        app: llm-service
    spec:
      containers:
      - name: llm-service
        image: your-registry/llm-service:latest
        env:
        - name: GENOPS_TEAM
          value: "production-team"
        - name: GENOPS_PROJECT
          value: "llm-service"
        - name: GENOPS_ENVIRONMENT
          value: "production"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: ai-api-keys
              key: openai-key
        - name: TRACELOOP_API_KEY
          valueFrom:
            secretKeyRef:
              name: observability-keys
              key: traceloop-key
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

### Enterprise Configuration

```python
# production_config.py
from genops.providers.traceloop import instrument_traceloop
from dataclasses import dataclass
from typing import List

@dataclass
class ProductionConfig:
    # Core settings
    team: str = "platform-engineering"
    project: str = "enterprise-llm"
    environment: str = "production"
    
    # Budget controls
    daily_budget_limit: float = 1000.0
    max_operation_cost: float = 50.0
    cost_approval_threshold: float = 100.0
    
    # High availability
    enable_ha: bool = True
    failover_regions: List[str] = None
    health_check_interval: int = 30
    
    # Compliance
    compliance_frameworks: List[str] = None
    audit_all_operations: bool = True
    
    def __post_init__(self):
        if self.failover_regions is None:
            self.failover_regions = ["us-west-2", "eu-west-1"]
        if self.compliance_frameworks is None:
            self.compliance_frameworks = ["SOC2", "GDPR", "HIPAA"]

# Initialize with production config
config = ProductionConfig()
adapter = instrument_traceloop(**config.__dict__)
```

---

## Performance & Scaling

### Performance Optimization

#### Sampling Configuration

```python
adapter = instrument_traceloop(
    team="high-volume-team",
    project="api-service",
    
    # Sampling for high-volume applications
    sampling_rate=0.1,          # Sample 10% of operations
    priority_sampling=True,      # Always sample high-priority operations
    error_sampling_rate=1.0,     # Always sample errors
    
    # Async processing
    enable_async_export=True,
    export_batch_size=100,
    export_timeout=5000,         # 5 seconds
    
    # Resource limits
    max_spans_per_operation=50,
    max_attribute_length=1000,
    max_events_per_span=100
)
```

#### Batch Processing

```python
# Batch operations for better performance
batch_requests = [
    "Request 1",
    "Request 2", 
    "Request 3"
]

with adapter.track_operation(
    operation_type="batch_processing",
    operation_name="bulk_analysis"
) as parent_span:
    
    results = []
    for i, request in enumerate(batch_requests):
        with adapter.track_operation(
            operation_type="individual_request",
            operation_name=f"request_{i}",
            parent_span=parent_span
        ) as child_span:
            # Process individual request
            result = process_request(request)
            results.append(result)
            
    parent_span.add_attributes({
        "batch.size": len(batch_requests),
        "batch.success_rate": len(results) / len(batch_requests)
    })
```

### Scaling Patterns

#### Circuit Breaker Pattern

```python
from genops.providers.traceloop import instrument_traceloop
import time

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise e

# Use with adapter
adapter = instrument_traceloop(
    team="resilient-team",
    project="circuit-breaker-demo"
)

circuit_breaker = CircuitBreaker()

with adapter.track_operation("protected_operation", "ai_call") as span:
    try:
        response = circuit_breaker.call(
            client.chat.completions.create,
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Protected call"}]
        )
        span.add_attributes({"circuit_breaker.state": "closed"})
    except Exception as e:
        span.add_attributes({
            "circuit_breaker.state": circuit_breaker.state,
            "circuit_breaker.failures": circuit_breaker.failure_count
        })
        raise
```

---

## Troubleshooting

### Common Issues

#### 1. Setup and Installation

**Issue**: `ModuleNotFoundError: No module named 'openllmetry'`

**Solution**:
```bash
pip install openllmetry
# Or reinstall with all dependencies
pip install genops[traceloop]
```

**Issue**: `No LLM provider API keys found`

**Solution**:
```bash
# Verify environment variables
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY

# Set API key if missing
export OPENAI_API_KEY="your-key-here"
```

#### 2. Governance and Cost Tracking

**Issue**: Cost attribution not visible in traces

**Solution**:
```python
# Ensure auto_instrument is called before LLM operations
from genops.providers.traceloop import auto_instrument

auto_instrument(team="your-team", project="your-project")

# Verify governance context
from genops.providers.traceloop import get_current_governance_context
context = get_current_governance_context()
print(context)  # Should show team, project, etc.
```

**Issue**: Policy violations not enforced

**Solution**:
```python
adapter = instrument_traceloop(
    team="your-team",
    project="your-project",
    governance_policy="enforced",  # Make sure this is set
    max_operation_cost=1.0         # Set appropriate limits
)
```

#### 3. Performance Issues

**Issue**: High governance overhead

**Solution**:
```python
# Enable sampling for high-volume applications
adapter = instrument_traceloop(
    team="your-team",
    project="your-project",
    sampling_rate=0.1,            # Sample only 10%
    enable_async_export=True,     # Async processing
    export_batch_size=100         # Batch exports
)
```

**Issue**: Memory usage growing over time

**Solution**:
```python
# Configure span limits
adapter = instrument_traceloop(
    team="your-team",
    project="your-project",
    max_spans_per_operation=50,   # Limit spans
    max_attribute_length=1000,    # Limit attribute size
    span_processor_queue_size=2048 # Limit queue size
)
```

### Debug Mode

Enable detailed debugging:

```python
import logging
import os

# Enable GenOps debug logging
os.environ["GENOPS_LOG_LEVEL"] = "DEBUG"
logging.basicConfig(level=logging.DEBUG)

# Enable OpenLLMetry debug logging
os.environ["OTEL_LOG_LEVEL"] = "debug"

# Run your application with detailed logs
adapter = instrument_traceloop(
    team="debug-team",
    project="debug-session"
)
```

### Validation and Health Checks

```python
# Comprehensive health check
from genops.providers.traceloop_validation import validate_setup, print_validation_result

result = validate_setup(
    include_connectivity_tests=True,
    include_performance_tests=True
)

print_validation_result(result, detailed=True)

# Check current governance status
from genops.providers.traceloop import get_budget_status, get_recent_operations_summary

budget_status = get_budget_status()
print(f"Budget status: {budget_status}")

operations = get_recent_operations_summary(limit=10)
print(f"Recent operations: {operations}")
```

---

## Migration & Compatibility

### Migrating from Other Observability Tools

#### From LangSmith

```python
# Before: LangSmith
from langsmith import traceable

@traceable
def my_function():
    return openai_client.chat.completions.create(...)

# After: GenOps + OpenLLMetry
from genops.providers.traceloop import auto_instrument

# Enable governance for existing code
auto_instrument(team="your-team", project="your-project")

# Your existing code works unchanged
@traceable  # Can keep existing decorators
def my_function():
    return openai_client.chat.completions.create(...)
    # Now includes governance automatically
```

#### From Weights & Biases

```python
# Before: W&B
import wandb

wandb.init(project="llm-tracking")
wandb.log({"cost": 0.001, "tokens": 100})

# After: GenOps + OpenLLMetry (automatic tracking)
from genops.providers.traceloop import instrument_traceloop

adapter = instrument_traceloop(
    team="your-team",
    project="llm-tracking"  # Same project name
)

with adapter.track_operation("llm_call", "tracked_operation") as span:
    response = openai_client.chat.completions.create(...)
    # Cost and tokens tracked automatically
```

### Compatibility with Existing OpenLLMetry

```python
# Existing OpenLLMetry code
from openllmetry.instrumentation.openai import OpenAIInstrumentor
from openllmetry.decorators import workflow

OpenAIInstrumentor().instrument()

@workflow(name="existing_workflow")
def existing_function():
    return openai_client.chat.completions.create(...)

# Add GenOps governance (no code changes needed)
from genops.providers.traceloop import auto_instrument

auto_instrument(team="your-team", project="your-project")

# Existing code now includes governance
result = existing_function()  # Enhanced with governance
```

### Compatibility Matrix

| Technology | Compatibility | Notes |
|------------|---------------|-------|
| **OpenLLMetry** | ✅ Full | Native integration, zero code changes |
| **Traceloop Platform** | ✅ Full | Optional commercial features |
| **OpenTelemetry** | ✅ Full | Industry-standard protocol |
| **Datadog** | ✅ Full | Native OTLP support |
| **Honeycomb** | ✅ Full | OpenTelemetry integration |
| **Grafana/Tempo** | ✅ Full | OTLP ingestion |
| **New Relic** | ✅ Full | OpenTelemetry support |
| **LangSmith** | ✅ Partial | Can coexist, OTLP export |
| **Weights & Biases** | ✅ Partial | Manual metric correlation |
| **MLflow** | ✅ Partial | Separate tracking systems |

---

## Advanced Use Cases

### Multi-Tenant Applications

```python
# Configure per-tenant governance
def create_tenant_adapter(tenant_id: str, tier: str):
    return instrument_traceloop(
        team=f"tenant-{tenant_id}",
        project="multi-tenant-llm",
        customer_id=tenant_id,
        
        # Tier-based budgets
        daily_budget_limit={
            "free": 10.0,
            "pro": 100.0, 
            "enterprise": 1000.0
        }.get(tier, 10.0),
        
        # Tier-based limits
        max_operation_cost={
            "free": 0.10,
            "pro": 1.0,
            "enterprise": 10.0
        }.get(tier, 0.10)
    )

# Use per-tenant
tenant_adapter = create_tenant_adapter("tenant-123", "enterprise")

with tenant_adapter.track_operation("customer_query", "support_request") as span:
    response = process_customer_request(request)
    span.add_attributes({
        "tenant.id": "tenant-123",
        "tenant.tier": "enterprise"
    })
```

### A/B Testing with Governance

```python
# A/B test with governance tracking
import random

def run_ab_test_with_governance(user_id: str, prompt: str):
    # Determine test variant
    variant = "control" if hash(user_id) % 2 == 0 else "treatment"
    
    adapter = instrument_traceloop(
        team="growth-team",
        project="prompt-optimization",
        customer_id=user_id
    )
    
    with adapter.track_operation(
        operation_type="ab_test",
        operation_name=f"prompt_test_{variant}",
        tags={
            "experiment": "prompt_optimization_v2",
            "variant": variant,
            "user_id": user_id
        }
    ) as span:
        
        # Use different prompts based on variant
        if variant == "control":
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=100
        )
        
        # Track experiment metadata
        span.add_attributes({
            "experiment.name": "prompt_optimization_v2",
            "experiment.variant": variant,
            "experiment.user_id": user_id,
            "response.length": len(response.choices[0].message.content)
        })
        
        return response.choices[0].message.content
```

---

For additional support and advanced configurations, refer to:
- [Quickstart Guide](../traceloop-quickstart.md)
- [Example Scripts](../examples/traceloop/)
- [API Documentation](../api/)
- [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)