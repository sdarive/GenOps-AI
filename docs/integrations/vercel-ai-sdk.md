# Vercel AI SDK Integration Guide

**Complete integration guide for GenOps governance with Vercel AI SDK across JavaScript/TypeScript and Python environments.**

## Overview

The Vercel AI SDK is a TypeScript toolkit for building AI-powered applications across React, Next.js, Vue, Svelte, and Node.js. GenOps provides comprehensive governance integration supporting:

- **20+ AI Providers**: OpenAI, Anthropic, Google, Cohere, Mistral, and more
- **All SDK Functions**: generateText, streamText, generateObject, embed, tool calling
- **Hybrid Integration**: JavaScript/Python bridge patterns
- **Production Ready**: Enterprise deployment, scaling, monitoring

## Quick Links

- **[5-Minute Quickstart](../vercel-ai-sdk-quickstart.md)** - Get started immediately
- **[Examples Suite](../../examples/vercel_ai_sdk/)** - 8 progressive examples
- **[API Reference](#api-reference)** - Complete API documentation

## Installation & Setup

### Prerequisites

- **Node.js 16+**: Required for Vercel AI SDK
- **Python 3.9+**: Required for GenOps integration
- **API Keys**: At least one AI provider API key

### Core Installation

```bash
# Python dependencies
pip install genops requests websockets aiohttp

# Node.js dependencies  
npm install ai @ai-sdk/openai @ai-sdk/anthropic @ai-sdk/google

# Optional: Additional providers
npm install @ai-sdk/cohere @ai-sdk/mistral
```

### Environment Configuration

```bash
# GenOps governance (required)
export GENOPS_TEAM="your-team"
export GENOPS_PROJECT="your-project"
export GENOPS_ENVIRONMENT="development"  # or staging/production

# AI provider API keys (at least one required)
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"

# Optional: Advanced configuration
export GENOPS_COST_CENTER="ai-department"
export GENOPS_CUSTOMER_ID="customer-123"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://your-collector:4317"
```

### Validation

```bash
# Quick validation
python -c "from genops.providers.vercel_ai_sdk_validation import quick_validation; print('✅ Ready!' if quick_validation() else '❌ Setup issues detected')"

# Comprehensive validation
python -c "from genops.providers.vercel_ai_sdk_validation import validate_setup; validate_setup()"
```

## Integration Patterns

### 1. Auto-Instrumentation Pattern (Recommended)

**Best for**: Existing applications, zero code changes required

```python
# Enable auto-instrumentation
from genops.providers.vercel_ai_sdk import auto_instrument

adapter = auto_instrument(
    team="ai-team",
    project="chatbot",
    environment="production"
)

# Generate instrumentation package
instrumentation_path = adapter.generate_instrumentation_code("./genops-instrumentation.js")
```

```javascript
// Use instrumented SDK (your existing code unchanged!)
const { generateText } = require('./genops-instrumentation');

const result = await generateText({
    model: 'gpt-4',
    prompt: 'Hello, world!'
});
// ✅ Automatic governance tracking added
```

### 2. Context Manager Pattern

**Best for**: Python-centric applications, detailed control

```python
from genops.providers.vercel_ai_sdk import GenOpsVercelAISDKAdapter

adapter = GenOpsVercelAISDKAdapter(
    integration_mode="subprocess",
    team="ai-team",
    project="data-analysis"
)

# Track specific operations
with adapter.track_request("generateText", "openai", "gpt-4") as request:
    # Execute your Vercel AI SDK JavaScript
    result = subprocess.run(["node", "your-script.js"])
    
    # Optionally update tracking
    request.input_tokens = 100
    request.output_tokens = 150
```

### 3. WebSocket Bridge Pattern

**Best for**: Real-time applications, streaming operations

```python
# Start WebSocket server
adapter = GenOpsVercelAISDKAdapter(
    integration_mode="websocket",
    websocket_port=8080
)
```

```javascript
// JavaScript client sends real-time telemetry
const { streamText } = require('ai');

const stream = await streamText({
    model: 'gpt-4',
    prompt: 'Write a story...',
    onChunk: (chunk) => {
        // Send telemetry to GenOps WebSocket server
        sendTelemetry({
            type: 'chunk',
            content: chunk,
            timestamp: Date.now()
        });
    }
});
```

## Core Functions Integration

### Text Generation

```python
# Python tracking wrapper
from genops.providers.vercel_ai_sdk import track_generate_text

with track_generate_text("openai", "gpt-4", 
                        team="content-team", 
                        project="blog-writer") as request:
    # Execute JavaScript
    result = execute_js_script("""
        const { generateText } = require('ai');
        const result = await generateText({
            model: 'gpt-4',
            prompt: 'Write a blog post about AI governance',
            maxTokens: 500,
            temperature: 0.7
        });
        console.log(JSON.stringify(result));
    """)
```

### Streaming Text

```python
# Real-time streaming with governance
with adapter.track_request("streamText", "anthropic", "claude-3-sonnet") as request:
    # Handle streaming chunks
    for chunk in stream_chunks:
        request.stream_chunks += 1
        # Real-time cost calculation per chunk
```

### Object Generation

```python
# Structured data with cost tracking
with track_generate_object("openai", "gpt-4",
                          operation_type="generateObject") as request:
    # Generate structured JSON
    result = execute_structured_generation()
```

### Embeddings

```python
# Embedding operations
with adapter.track_request("embed", "openai", "text-embedding-ada-002") as request:
    # Track embedding costs
    embeddings = generate_embeddings(["text1", "text2", "text3"])
    request.input_tokens = len(embeddings) * 100  # Estimate
```

### Tool Calling & Agents

```python
# Complex agent workflows
with adapter.track_request("agent_workflow", "openai", "gpt-4") as request:
    # Track tool usage
    request.tools_used = ["web_search", "calculator", "database_query"]
    
    # Execute agent workflow
    result = run_agent_workflow()
```

## Multi-Provider Configuration

### Provider Setup

```javascript
// Configure multiple providers
const { openai } = require('@ai-sdk/openai');
const { anthropic } = require('@ai-sdk/anthropic');
const { google } = require('@ai-sdk/google');

// GenOps automatically tracks all providers
const providers = {
    fast: openai('gpt-3.5-turbo'),      // Fast & cheap
    smart: anthropic('claude-3-opus'),   // High quality
    vision: google('gemini-pro-vision')  // Multimodal
};
```

### Cost Optimization Patterns

```python
# Cost-aware provider selection
from genops.providers.vercel_ai_sdk_pricing import estimate_cost

def select_optimal_provider(prompt, budget_limit):
    providers = [
        ("openai", "gpt-3.5-turbo"),
        ("anthropic", "claude-3-haiku"),
        ("google", "gemini-pro")
    ]
    
    for provider, model in providers:
        min_cost, max_cost = estimate_cost(
            provider, model, len(prompt), 200
        )
        if max_cost <= budget_limit:
            return provider, model
    
    raise ValueError("No provider within budget")
```

## Production Deployment

### Docker Integration

```dockerfile
# Multi-stage build for Node.js + Python
FROM node:18-alpine AS node-stage
WORKDIR /app
COPY package*.json ./
RUN npm install

FROM python:3.11-slim AS python-stage
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Final stage
FROM python:3.11-slim
COPY --from=node-stage /usr/local/bin/node /usr/local/bin/
COPY --from=node-stage /usr/local/lib/node_modules/ /usr/local/lib/node_modules/
COPY --from=node-stage /app/node_modules ./node_modules
COPY --from=python-stage /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application code
COPY . .

# Environment variables
ENV GENOPS_TEAM=production
ENV GENOPS_ENVIRONMENT=production
ENV OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:14268/api/traces

CMD ["python", "app.py"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vercel-ai-app
  labels:
    app: vercel-ai-app
    genops.ai/instrumented: "true"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vercel-ai-app
  template:
    metadata:
      labels:
        app: vercel-ai-app
    spec:
      containers:
      - name: app
        image: your-registry/vercel-ai-app:latest
        env:
        # GenOps Configuration
        - name: GENOPS_TEAM
          value: "production"
        - name: GENOPS_PROJECT
          value: "ai-service"
        - name: GENOPS_ENVIRONMENT
          value: "production"
        
        # OpenTelemetry Configuration
        - name: OTEL_EXPORTER_OTLP_ENDPOINT
          value: "http://jaeger-collector:14268/api/traces"
        - name: OTEL_SERVICE_NAME
          value: "vercel-ai-service"
        
        # AI Provider Keys (from secrets)
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: ai-provider-keys
              key: openai-api-key
        
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        
        # Health checks
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Secret
metadata:
  name: ai-provider-keys
type: Opaque
data:
  openai-api-key: <base64-encoded-key>
  anthropic-api-key: <base64-encoded-key>
```

### CI/CD Integration

```yaml
# GitHub Actions workflow
name: Deploy Vercel AI SDK App

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '18'
        
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        npm install
        pip install genops
    
    - name: Validate GenOps integration
      run: |
        python -c "from genops.providers.vercel_ai_sdk_validation import validate_setup; assert validate_setup(verbose=False).all_passed"
      env:
        GENOPS_TEAM: ci-testing
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    
    - name: Run tests with governance
      run: |
        # Tests automatically include GenOps telemetry
        npm test
        python -m pytest tests/
    
    - name: Build and deploy
      run: |
        docker build -t vercel-ai-app:${{ github.sha }} .
        docker push your-registry/vercel-ai-app:${{ github.sha }}
```

## Performance & Scaling

### Performance Characteristics

- **Telemetry Overhead**: <5ms per request
- **Memory Usage**: ~10MB for adapter instance
- **Network Overhead**: Batched OTLP export (configurable)
- **CPU Impact**: Minimal (<1% additional CPU usage)

### High-Volume Configuration

```python
# Optimized for high-volume applications
adapter = GenOpsVercelAISDKAdapter(
    # Use subprocess mode for better isolation
    integration_mode="subprocess",
    
    # Batch telemetry exports
    batch_size=100,
    batch_timeout=30,
    
    # Sample for high-volume (10% sampling)
    sampling_rate=0.1,
    
    # Async telemetry export
    async_export=True
)

# Configure OpenTelemetry sampling
import os
os.environ['OTEL_TRACES_SAMPLER'] = 'traceidratio'
os.environ['OTEL_TRACES_SAMPLER_ARG'] = '0.1'  # 10% sampling
```

### Scaling Patterns

```python
# Circuit breaker for external dependencies
from genops.providers.vercel_ai_sdk import GenOpsVercelAISDKAdapter

class ResilientAdapter(GenOpsVercelAISDKAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60
        )
    
    def track_request(self, *args, **kwargs):
        if self.circuit_breaker.is_open():
            # Graceful degradation - minimal tracking
            return self.minimal_tracking_context(*args, **kwargs)
        
        try:
            return super().track_request(*args, **kwargs)
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise
```

## Monitoring & Observability

### Dashboard Integration

#### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "GenOps Vercel AI SDK Monitoring",
    "panels": [
      {
        "title": "AI Request Rate",
        "targets": [
          {
            "expr": "rate(genops_vercel_ai_sdk_requests_total[5m])",
            "legendFormat": "{{provider}} - {{model}}"
          }
        ]
      },
      {
        "title": "Cost per Hour",
        "targets": [
          {
            "expr": "increase(genops_cost_total[1h])",
            "legendFormat": "{{team}} - {{project}}"
          }
        ]
      },
      {
        "title": "Token Usage",
        "targets": [
          {
            "expr": "genops_tokens_input_total + genops_tokens_output_total",
            "legendFormat": "{{model}}"
          }
        ]
      }
    ]
  }
}
```

#### Datadog Dashboard

```python
# Datadog integration example
from datadog import initialize, api

# Custom metrics for Vercel AI SDK
def send_custom_metrics(request_data):
    api.Metric.send(
        metric='genops.vercel_ai_sdk.cost',
        points=[(time.time(), request_data.cost)],
        tags=[
            f"team:{request_data.governance_attrs['team']}",
            f"provider:{request_data.provider}",
            f"model:{request_data.model}"
        ]
    )
```

### Alerting Rules

```yaml
# Prometheus alerting rules
groups:
- name: genops_vercel_ai_sdk
  rules:
  - alert: HighAICost
    expr: increase(genops_cost_total[1h]) > 100
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High AI costs detected"
      description: "AI costs exceeded $100/hour for team {{ $labels.team }}"
  
  - alert: AIRequestFailures
    expr: rate(genops_vercel_ai_sdk_errors_total[5m]) > 0.1
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High AI request failure rate"
      description: "AI request failure rate is {{ $value }} for provider {{ $labels.provider }}"
```

## API Reference

### GenOpsVercelAISDKAdapter

```python
class GenOpsVercelAISDKAdapter:
    def __init__(
        self,
        integration_mode: str = "python_wrapper",  # or "websocket", "subprocess"
        websocket_port: int = 8080,
        team: Optional[str] = None,
        project: Optional[str] = None,
        environment: Optional[str] = None,
        cost_center: Optional[str] = None,
        customer_id: Optional[str] = None,
        feature: Optional[str] = None,
        **kwargs
    ):
        """Initialize Vercel AI SDK adapter with governance."""
    
    def track_request(
        self,
        operation_type: str,
        provider: str,
        model: str,
        **kwargs
    ) -> ContextManager[VercelAISDKRequest]:
        """Track a Vercel AI SDK request with governance."""
    
    def generate_instrumentation_code(
        self,
        output_path: str = "./genops-vercel-instrumentation.js"
    ) -> str:
        """Generate JavaScript instrumentation code."""
```

### Auto-Instrumentation Functions

```python
def auto_instrument(
    integration_mode: str = "python_wrapper",
    team: Optional[str] = None,
    project: Optional[str] = None,
    **kwargs
) -> GenOpsVercelAISDKAdapter:
    """Auto-instrument Vercel AI SDK applications."""

def track_generate_text(provider: str, model: str, **kwargs):
    """Convenience function for tracking generateText operations."""

def track_stream_text(provider: str, model: str, **kwargs):
    """Convenience function for tracking streamText operations."""

def track_generate_object(provider: str, model: str, **kwargs):
    """Convenience function for tracking generateObject operations."""

def track_embed(provider: str, model: str, **kwargs):
    """Convenience function for tracking embed operations."""
```

### Validation Functions

```python
def validate_setup(
    check_nodejs: bool = True,
    check_npm_packages: bool = True,
    check_python_deps: bool = True,
    check_environment: bool = True,
    check_genops_config: bool = True,
    check_provider_access: bool = False,
    verbose: bool = True
) -> SetupValidationSummary:
    """Comprehensive setup validation."""

def quick_validation() -> bool:
    """Quick validation check."""

def print_validation_result(result: SetupValidationSummary) -> None:
    """Print validation results."""
```

### Pricing Functions

```python
def calculate_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int
) -> CostBreakdown:
    """Calculate cost for a request."""

def estimate_cost(
    provider: str,
    model: str,
    prompt_length: int,
    response_length: int = None
) -> Tuple[Decimal, Decimal]:
    """Estimate cost before making request."""

def get_model_info(provider: str, model: str) -> Optional[ModelPricing]:
    """Get model information and capabilities."""

def get_supported_providers() -> Dict[str, List[str]]:
    """Get list of supported providers and models."""
```

## Advanced Use Cases

### Multi-Tenant SaaS

```python
# Customer-specific governance
def create_customer_adapter(customer_id: str, plan: str):
    return GenOpsVercelAISDKAdapter(
        team=f"customer-{customer_id}",
        project="saas-platform",
        customer_id=customer_id,
        cost_center=f"customer-revenue-{plan}",
        
        # Plan-specific budget limits
        budget_limit=get_budget_for_plan(plan),
        
        # Custom sampling for different plans
        sampling_rate=get_sampling_rate(plan)
    )

# Usage in SaaS application
customer_adapter = create_customer_adapter("cust-123", "enterprise")
with customer_adapter.track_request("generateText", "openai", "gpt-4") as request:
    # Customer-isolated tracking
    result = generate_for_customer(customer_id, prompt)
```

### Enterprise Budget Controls

```python
# Budget enforcement
class BudgetEnforcedAdapter(GenOpsVercelAISDKAdapter):
    def __init__(self, *args, monthly_budget: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.monthly_budget = monthly_budget
    
    def track_request(self, *args, **kwargs):
        # Check budget before request
        current_spend = self.get_monthly_spend()
        if current_spend >= self.monthly_budget:
            raise BudgetExceededException(
                f"Monthly budget ${self.monthly_budget} exceeded. "
                f"Current spend: ${current_spend}"
            )
        
        return super().track_request(*args, **kwargs)
```

### A/B Testing Integration

```python
# A/B testing with governance
def ab_test_models(prompt: str, user_id: str):
    # Determine test group
    test_group = hash(user_id) % 2
    
    if test_group == 0:
        # Control group
        with track_generate_text("openai", "gpt-3.5-turbo",
                                feature="control-group") as request:
            return generate_text_openai(prompt)
    else:
        # Test group
        with track_generate_text("anthropic", "claude-3-haiku",
                                feature="test-group") as request:
            return generate_text_anthropic(prompt)
```

## Migration Guide

### From Direct Vercel AI SDK

**Before (Direct SDK):**
```javascript
import { generateText } from 'ai';
import { openai } from '@ai-sdk/openai';

const result = await generateText({
    model: openai('gpt-4'),
    prompt: 'Hello'
});
```

**After (With GenOps):**
```python
# Python wrapper approach
from genops.providers.vercel_ai_sdk import auto_instrument

adapter = auto_instrument(team="your-team")
instrumentation_path = adapter.generate_instrumentation_code()
```

```javascript
// Use generated instrumentation (code unchanged!)
import { generateText } from './genops-vercel-instrumentation';

const result = await generateText({
    model: 'gpt-4',  // Simplified model syntax
    prompt: 'Hello'
});
// ✅ Now includes governance tracking
```

### Migration Checklist

- [ ] Install GenOps: `pip install genops`
- [ ] Set governance environment variables
- [ ] Run validation: `validate_setup()`
- [ ] Generate instrumentation code
- [ ] Update import statements to use instrumentation
- [ ] Verify telemetry export in observability dashboard
- [ ] Set up alerting and monitoring
- [ ] Document team-specific governance attributes

## Troubleshooting

### Common Issues

#### "Node.js not found"
```bash
# Install Node.js via nvm (recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 18
nvm use 18

# Or direct install: https://nodejs.org/
```

#### "Vercel AI SDK not installed"
```bash
npm install ai @ai-sdk/openai
# For other providers:
npm install @ai-sdk/anthropic @ai-sdk/google
```

#### "WebSocket connection failed"
```bash
# Check port availability
netstat -an | grep 8080

# Try different port
export GENOPS_WEBSOCKET_PORT=8081
```

#### "Cost calculation errors"
```bash
# Update provider pricing data
pip install --upgrade genops

# Check provider calculator availability
python -c "from genops.providers.vercel_ai_sdk_pricing import get_supported_providers; print(get_supported_providers())"
```

#### "Telemetry not appearing in dashboard"
```bash
# Check OpenTelemetry configuration
echo $OTEL_EXPORTER_OTLP_ENDPOINT

# Verify collector connectivity
curl -v $OTEL_EXPORTER_OTLP_ENDPOINT/v1/traces

# Enable debug logging
export OTEL_LOG_LEVEL=debug
```

### Debug Mode

```python
# Enable comprehensive debugging
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable OpenTelemetry debug
import os
os.environ['OTEL_LOG_LEVEL'] = 'debug'

# Run with detailed validation
from genops.providers.vercel_ai_sdk_validation import validate_setup
result = validate_setup(verbose=True, check_provider_access=True)
```

## Support & Community

### Getting Help

- **Documentation**: This guide and [quickstart](../vercel-ai-sdk-quickstart.md)
- **Examples**: [Progressive examples suite](../../examples/vercel_ai_sdk/)
- **Issues**: [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)
- **Community**: [Discord/Slack community](#)

### Contributing

- **Code Contributions**: Follow [CONTRIBUTING.md](../../CONTRIBUTING.md)
- **Documentation**: Help improve guides and examples
- **Testing**: Add test cases and integration scenarios
- **Feedback**: Share usage patterns and improvement suggestions

### Roadmap

**Coming Soon:**
- [ ] React Server Components integration
- [ ] Edge Runtime support
- [ ] More streaming optimizations
- [ ] Advanced cost optimization algorithms
- [ ] Built-in A/B testing utilities

**Long Term:**
- [ ] Visual workflow builder integration
- [ ] Advanced governance policy engine
- [ ] Machine learning cost prediction
- [ ] Multi-region deployment patterns

---

**Next Steps**: Try the [5-minute quickstart](../vercel-ai-sdk-quickstart.md) or explore [progressive examples](../../examples/vercel_ai_sdk/)