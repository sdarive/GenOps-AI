# GenOps-AI Architecture: Detailed Component Internals

## Overview
This diagram provides a deep dive into GenOps-AI's internal architecture, showing how components interact to provide instrumentation, cost tracking, policy enforcement, and telemetry export.

## Detailed Component Architecture

```mermaid
graph TB
    subgraph "Application Layer"
        APP[Application Code<br/>LangChain, OpenAI, etc.]
    end

    subgraph "GenOps-AI SDK"

        subgraph "Public API Interface"
            INIT[GenOps.init&#40;&#41;<br/>Bootstrap & Auto-detection]
            TRACK_DEC[@track_usage&#40;&#41;<br/>Function Decorator]
            POLICY_DEC[@enforce_policy&#40;&#41;<br/>Policy Decorator]
            TRACK_CTX[track&#40;&#41; Context Manager<br/>Block-level Tracking]
            MANUAL[Manual APIs<br/>track_cost, record_eval]
        end

        subgraph "Core Module (core/)"
            INST[Instrumentation Engine<br/>Span Creation & Management]
            COST_CALC[Cost Calculator<br/>Provider Pricing Tables]
            POLICY_ENG[Policy Engine<br/>Rule Evaluation]
            EVAL_ENG[Evaluation Engine<br/>Metrics Collection]
            CTX_PROP[Context Propagator<br/>Team/Project Attribution]
        end

        subgraph "Config Module (config/)"
            CFG_LOAD[Configuration Loader<br/>ENV, YAML, CLI args]
            POLICY_REG[Policy Registry<br/>In-memory Policy Store]
            PROVIDER_CFG[Provider Configuration<br/>API Keys, Endpoints]
            SAMPLING[Sampling Configuration<br/>Rate Limits, Filters]
        end

        subgraph "Providers Module (providers/)"
            AUTO_DETECT[Auto-Detection<br/>Scan sys.modules]

            subgraph "LLM Providers"
                OAI[OpenAI Adapter<br/>Client Wrapper]
                ANT[Anthropic Adapter<br/>Client Wrapper]
                BED[AWS Bedrock Adapter<br/>boto3 Wrapper]
                LLM[LiteLLM Adapter<br/>Universal Proxy]
            end

            subgraph "Framework Providers"
                LC_PROV[LangChain Provider<br/>Callback Handler]
                LI_PROV[LlamaIndex Provider<br/>Event Handler]
                HS_PROV[Haystack Provider<br/>Pipeline Hook]
            end

            PROV_REGISTRY[Provider Registry<br/>Active Providers Map]
        end

        subgraph "Processors Module (processors/)"
            ENRICH[Enrichment Processor<br/>Add Metadata]
            FILTER[Filter Processor<br/>PII Removal, Sampling]
            TRANSFORM[Transform Processor<br/>Attribute Mapping]
            AGGR[Aggregation Processor<br/>Batch Metrics]
        end

        subgraph "Exporters Module (exporters/)"
            OTLP_EXP[OTLP Exporter<br/>gRPC/HTTP]
            CONSOLE_EXP[Console Exporter<br/>Debug Output]
            JAEGER_EXP[Jaeger Exporter<br/>Native Protocol]
            DD_EXP[Datadog Exporter<br/>Agent API]
            CUSTOM_EXP[Custom Exporters<br/>User-defined]

            EXP_FACTORY[Exporter Factory<br/>Multi-destination Routing]
        end

        subgraph "CLI Module (cli/)"
            CLI_INIT[genops init<br/>Project Setup]
            CLI_VAL[genops validate<br/>Config Validation]
            CLI_TEST[genops test<br/>Connection Testing]
            CLI_EXPORT[genops export<br/>Telemetry Export]
        end

        subgraph "Auto-Instrumentation (auto_instrumentation.py)"
            MONKEY[Monkey Patching Engine<br/>Runtime Code Injection]
            IMPORT_HOOK[Import Hook<br/>Module Load Interception]
            WRAP[Wrapper Generator<br/>Dynamic Proxy Creation]
        end

    end

    subgraph "OpenTelemetry SDK"
        OTEL_TRACE[Tracer Provider<br/>Span Management]
        OTEL_METRIC[Meter Provider<br/>Metrics Collection]
        OTEL_RES[Resource<br/>Service Identity]
        OTEL_BATCH[Batch Processor<br/>Performance]
    end

    subgraph "External Systems"
        OBS[Observability Platforms<br/>Datadog, Grafana, etc.]
        AI_API[AI Provider APIs<br/>OpenAI, Anthropic, etc.]
    end

    %% Application to API Interface
    APP --> INIT
    APP --> TRACK_DEC
    APP --> POLICY_DEC
    APP --> TRACK_CTX
    APP --> MANUAL

    %% API Interface to Core
    INIT --> AUTO_DETECT
    TRACK_DEC --> INST
    POLICY_DEC --> POLICY_ENG
    TRACK_CTX --> INST
    MANUAL --> COST_CALC
    MANUAL --> EVAL_ENG

    %% Auto-detection to Providers
    AUTO_DETECT --> MONKEY
    AUTO_DETECT --> IMPORT_HOOK
    MONKEY --> WRAP
    IMPORT_HOOK --> WRAP

    %% Wrapper to Provider Adapters
    WRAP --> OAI
    WRAP --> ANT
    WRAP --> BED
    WRAP --> LLM
    WRAP --> LC_PROV
    WRAP --> LI_PROV
    WRAP --> HS_PROV

    %% Provider Registry
    OAI --> PROV_REGISTRY
    ANT --> PROV_REGISTRY
    BED --> PROV_REGISTRY
    LLM --> PROV_REGISTRY
    LC_PROV --> PROV_REGISTRY
    LI_PROV --> PROV_REGISTRY
    HS_PROV --> PROV_REGISTRY

    %% Config Module Interactions
    CFG_LOAD --> PROVIDER_CFG
    CFG_LOAD --> POLICY_REG
    CFG_LOAD --> SAMPLING

    PROVIDER_CFG --> PROV_REGISTRY
    POLICY_REG --> POLICY_ENG
    SAMPLING --> FILTER

    %% Core to Core
    INST --> CTX_PROP
    CTX_PROP --> COST_CALC
    POLICY_ENG --> INST

    %% Core to Processors
    INST --> ENRICH
    COST_CALC --> ENRICH
    EVAL_ENG --> ENRICH

    ENRICH --> FILTER
    FILTER --> TRANSFORM
    TRANSFORM --> AGGR

    %% Processors to Exporters
    AGGR --> EXP_FACTORY
    EXP_FACTORY --> OTLP_EXP
    EXP_FACTORY --> CONSOLE_EXP
    EXP_FACTORY --> JAEGER_EXP
    EXP_FACTORY --> DD_EXP
    EXP_FACTORY --> CUSTOM_EXP

    %% Exporters to OpenTelemetry
    OTLP_EXP --> OTEL_BATCH
    JAEGER_EXP --> OTEL_BATCH
    DD_EXP --> OTEL_BATCH

    %% OpenTelemetry Integration
    INST --> OTEL_TRACE
    EVAL_ENG --> OTEL_METRIC
    CTX_PROP --> OTEL_RES

    OTEL_TRACE --> OTEL_BATCH
    OTEL_METRIC --> OTEL_BATCH

    %% External Systems
    OTEL_BATCH --> OBS
    PROV_REGISTRY --> AI_API

    %% CLI Interactions
    CLI_INIT --> CFG_LOAD
    CLI_VAL --> CFG_LOAD
    CLI_TEST --> PROV_REGISTRY
    CLI_EXPORT --> EXP_FACTORY

    %% Styling
    classDef appStyle fill:#e1f5ff,stroke:#0066cc,stroke-width:2px
    classDef apiStyle fill:#fff4e6,stroke:#ff9800,stroke-width:2px
    classDef coreStyle fill:#e8f5e9,stroke:#4caf50,stroke-width:3px
    classDef configStyle fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
    classDef providerStyle fill:#fce4ec,stroke:#e91e63,stroke-width:2px
    classDef processorStyle fill:#fff9c4,stroke:#fbc02d,stroke-width:2px
    classDef exporterStyle fill:#e0f2f1,stroke:#009688,stroke-width:2px
    classDef cliStyle fill:#fbe9e7,stroke:#ff5722,stroke-width:2px
    classDef autoStyle fill:#f1f8e9,stroke:#8bc34a,stroke-width:2px
    classDef otelStyle fill:#ede7f6,stroke:#673ab7,stroke-width:2px
    classDef externalStyle fill:#eceff1,stroke:#607d8b,stroke-width:2px

    class APP appStyle
    class INIT,TRACK_DEC,POLICY_DEC,TRACK_CTX,MANUAL apiStyle
    class INST,COST_CALC,POLICY_ENG,EVAL_ENG,CTX_PROP coreStyle
    class CFG_LOAD,POLICY_REG,PROVIDER_CFG,SAMPLING configStyle
    class AUTO_DETECT,OAI,ANT,BED,LLM,LC_PROV,LI_PROV,HS_PROV,PROV_REGISTRY providerStyle
    class ENRICH,FILTER,TRANSFORM,AGGR processorStyle
    class OTLP_EXP,CONSOLE_EXP,JAEGER_EXP,DD_EXP,CUSTOM_EXP,EXP_FACTORY exporterStyle
    class CLI_INIT,CLI_VAL,CLI_TEST,CLI_EXPORT cliStyle
    class MONKEY,IMPORT_HOOK,WRAP autoStyle
    class OTEL_TRACE,OTEL_METRIC,OTEL_RES,OTEL_BATCH otelStyle
    class OBS,AI_API externalStyle
```

## Component Deep Dive

### 1. **Public API Interface**

The surface area developers interact with:

**`GenOps.init()`**:
```python
def init(
    service_name: str = "my-ai-service",
    exporters: List[str] = ["otlp"],
    auto_instrument: bool = True,
    config_path: Optional[str] = None
) -> None:
    """
    Bootstrap GenOps-AI instrumentation
    - Loads configuration from ENV/YAML/args
    - Auto-detects installed AI libraries
    - Registers providers and exporters
    - Initializes OpenTelemetry SDK
    """
```

**`@track_usage()` Decorator**:
```python
@track_usage(
    operation: str,
    team: str,
    project: str,
    metadata: Dict[str, Any] = {}
)
def your_function():
    """
    Creates OpenTelemetry span around function
    - Captures duration, errors, return values
    - Adds team/project attributes
    - Integrates with cost calculator
    """
```

**`@enforce_policy()` Decorator**:
```python
@enforce_policy(policies: List[Policy])
def your_function():
    """
    Pre-execution policy validation
    - Checks budget limits before call
    - Validates allowed models/providers
    - Blocks or warns based on configuration
    """
```

**`track()` Context Manager**:
```python
with track(operation="rag_query", team="search") as span:
    span.set_attribute("query_type", "semantic")
    span.set_attribute("index_size", 1000000)
    # Operations tracked within block
```

### 2. **Core Module** (`core/`)

The brain of GenOps-AI:

**Instrumentation Engine**:
- Creates OpenTelemetry spans for tracked operations
- Manages span lifecycle (start, end, error handling)
- Injects context (trace ID, span ID) for distributed tracing
- Handles nested spans for complex operations

**Cost Calculator**:
```python
class CostCalculator:
    """
    Provider-specific pricing logic

    Pricing tables embedded:
    - OpenAI: $0.03/1K input, $0.06/1K output (GPT-4)
    - Anthropic: $0.015/1K input, $0.075/1K output (Claude Opus)
    - Bedrock: Variable by model

    Calculates:
    - Input token cost
    - Output token cost
    - Total cost in USD
    - Currency conversion (future)
    """
```

**Policy Engine**:
```python
class PolicyEngine:
    """
    Rule evaluation system

    Policy Types:
    1. BudgetPolicy: Daily/monthly spend limits
    2. ModelPolicy: Allowed/blocked models
    3. TeamPolicy: Team-based restrictions
    4. RateLimitPolicy: Request throttling
    5. ContentPolicy: PII/sensitive data detection

    Enforcement Modes:
    - BLOCKED: Raise exception, prevent execution
    - WARNING: Log violation, allow execution
    - AUDIT: Record only, no alerts
    """
```

**Evaluation Engine**:
- Records quality metrics (ROUGE, BLEU, custom)
- Tracks latency percentiles (p50, p95, p99)
- Monitors token efficiency (tokens/second)
- Aggregates evaluation scores

**Context Propagator**:
- Manages W3C Trace Context headers
- Propagates team/project across service boundaries
- Handles baggage for custom attributes
- Ensures consistent attribution in distributed systems

### 3. **Config Module** (`config/`)

Configuration management with precedence:

**Priority Order**:
1. CLI arguments (highest)
2. Environment variables
3. YAML configuration file
4. Default values (lowest)

**Configuration Schema**:
```yaml
genops:
  service_name: "my-ai-service"
  environment: "production"

  exporters:
    - type: "otlp"
      endpoint: "https://otel-collector:4317"
      headers:
        x-api-key: "${DATADOG_API_KEY}"

    - type: "console"
      enabled: true  # For debugging

  providers:
    openai:
      api_key: "${OPENAI_API_KEY}"
      track_embeddings: true

    anthropic:
      api_key: "${ANTHROPIC_API_KEY}"
      max_retries: 3

  policies:
    - name: "team_budget"
      type: "budget"
      limit: 1000.0  # USD
      period: "daily"
      teams: ["data-science"]
      enforcement: "BLOCKED"

    - name: "production_models"
      type: "model"
      allowed: ["gpt-4", "claude-opus-4"]
      environment: "production"
      enforcement: "WARNING"

  sampling:
    rate: 1.0  # 100% sampling
    max_spans_per_second: 1000
```

**Policy Registry**:
- In-memory store of active policies
- Fast O(1) lookup by policy name
- Thread-safe for concurrent access
- Runtime policy updates (advanced)

### 4. **Providers Module** (`providers/`)

The integration layer for 30+ AI providers:

**Auto-Detection Logic**:
```python
def auto_detect_providers() -> List[str]:
    """
    Scan sys.modules for AI libraries

    Detection patterns:
    - openai in sys.modules → OpenAI detected
    - anthropic in sys.modules → Anthropic detected
    - langchain in sys.modules → LangChain detected

    Returns list of detected provider names
    """
```

**Provider Adapter Pattern**:
```python
class OpenAIAdapter:
    """
    Wraps OpenAI client methods

    Instrumented methods:
    - chat.completions.create()
    - completions.create()
    - embeddings.create()

    Captures:
    - Model name (gpt-4, gpt-3.5-turbo)
    - Token counts (prompt_tokens, completion_tokens)
    - Latency (time to first token, total time)
    - Cost (calculated from tokens × pricing)
    - Errors (API errors, rate limits)
    """

    def instrument_client(self, client: OpenAI) -> OpenAI:
        """
        Monkey-patch client methods
        Original functionality preserved
        Telemetry layer added transparently
        """
```

**LangChain Provider**:
```python
class GenOpsCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback integration

    Hooks:
    - on_llm_start: Create span for LLM call
    - on_llm_end: Record tokens, cost, latency
    - on_llm_error: Capture error details
    - on_chain_start: Track chain execution
    - on_tool_start: Monitor tool usage
    """
```

### 5. **Processors Module** (`processors/`)

Data pipeline for telemetry processing:

**Enrichment Processor**:
```python
def enrich_span(span: Span) -> Span:
    """
    Add computed attributes:
    - genops.cost.total (from tokens × pricing)
    - genops.cost.currency ("USD")
    - genops.tokens.total (input + output)
    - genops.latency.ttft (time to first token)
    - genops.model.family (gpt-4 → openai-chat)
    - genops.environment (from config)
    """
```

**Filter Processor**:
```python
def filter_span(span: Span, config: FilterConfig) -> Optional[Span]:
    """
    PII Removal:
    - Remove prompt/response content by default
    - Redact email addresses, phone numbers
    - Mask API keys in error messages

    Sampling:
    - Probabilistic sampling (rate = 0.1 → 10%)
    - Head sampling (before expensive processing)
    - Tail sampling (after latency known)

    Returns None if span should be dropped
    """
```

**Transform Processor**:
```python
def transform_attributes(span: Span) -> Span:
    """
    Normalize attribute names across providers:
    - openai.tokens.prompt → genops.tokens.input
    - anthropic.usage.input_tokens → genops.tokens.input
    - bedrock.inputTokenCount → genops.tokens.input

    Platform-specific mappings for compatibility
    """
```

**Aggregation Processor**:
```python
def aggregate_metrics(spans: List[Span]) -> Metrics:
    """
    Batch aggregation for efficiency:
    - Sum total costs per team/project
    - Calculate token throughput
    - Compute latency percentiles
    - Count requests by model

    Reduces telemetry volume 10-100x
    """
```

### 6. **Exporters Module** (`exporters/`)

Output adapters for observability platforms:

**OTLP Exporter** (Primary):
```python
class OTLPExporter:
    """
    Standard OpenTelemetry Protocol

    Protocols:
    - gRPC (port 4317) - Production, efficient
    - HTTP (port 4318) - Development, debugging

    Features:
    - Automatic retry with exponential backoff
    - Compression (gzip)
    - TLS encryption
    - Authentication headers

    Compatible with:
    - OpenTelemetry Collector
    - Datadog Agent (OTLP receiver)
    - Grafana Tempo
    - Honeycomb
    - Any OTLP-compatible backend
    """
```

**Datadog Exporter** (Native):
```python
class DatadogExporter:
    """
    Direct Datadog Agent API

    Advantages over OTLP:
    - Native Datadog span format
    - Better dashboard integration
    - Lower latency (no collector)

    API: POST /v0.4/traces
    """
```

**Console Exporter** (Debug):
```python
class ConsoleExporter:
    """
    Pretty-printed JSON to stdout

    Use cases:
    - Local development
    - CI/CD validation
    - Debugging instrumentation

    Format:
    {
      "trace_id": "abc123",
      "span_name": "openai.chat.completions",
      "duration_ms": 1234,
      "attributes": {
        "genops.cost.total": 0.002,
        "genops.tokens.input": 8,
        "genops.tokens.output": 12
      }
    }
    """
```

**Exporter Factory**:
```python
def create_exporters(config: Config) -> List[Exporter]:
    """
    Multi-destination routing:
    - Send to Datadog AND Grafana simultaneously
    - Fan-out pattern for redundancy
    - Per-exporter error isolation
    """
```

### 7. **CLI Module** (`cli/`)

Command-line tools for operations:

**`genops init`**:
```bash
genops init --service-name my-ai-service --exporter datadog
# Creates genops.yaml config file
# Detects installed AI libraries
# Validates API keys
# Generates example code
```

**`genops validate`**:
```bash
genops validate --config genops.yaml
# Checks YAML syntax
# Validates policy definitions
# Tests exporter connections
# Reports configuration errors
```

**`genops test`**:
```bash
genops test --provider openai
# Makes test API call
# Validates instrumentation
# Checks telemetry export
# Reports round-trip success
```

**`genops export`**:
```bash
genops export --format json --output spans.json
# Exports collected telemetry
# For offline analysis
# Debugging, auditing
```

### 8. **Auto-Instrumentation** (`auto_instrumentation.py`)

The magic behind zero-code setup:

**Import Hook**:
```python
class GenOpsImportHook:
    """
    sys.meta_path hook for import interception

    When Python imports a module:
    1. Hook checks if it's an AI library
    2. If yes, applies instrumentation
    3. Original module returned with wrappers

    Example:
    import openai  # GenOps intercepts
    # openai.ChatCompletion.create is now wrapped
    """
```

**Monkey Patching**:
```python
def monkey_patch_openai():
    """
    Runtime modification of OpenAI library

    Original: openai.ChatCompletion.create = original_function
    Patched:  openai.ChatCompletion.create = wrapper(original_function)

    Wrapper:
    1. Start GenOps span
    2. Call original function
    3. Capture result
    4. Calculate cost/tokens
    5. End span
    6. Return result to caller
    """
```

**Dynamic Proxy**:
```python
class InstrumentedClient:
    """
    Proxy pattern for client wrapping

    All method calls intercepted:
    - client.chat.completions.create(...)
      → proxy.intercept("chat.completions.create", ...)
      → add_telemetry()
      → real_client.chat.completions.create(...)
      → capture_response()
      → return to caller
    """
```

## Data Flow: Request Lifecycle

### Step-by-Step Execution

```
1. Developer Code Execution
   ↓
2. Decorator/Context Manager Intercepts
   ↓
3. Policy Engine Validates (pre-execution)
   → If BLOCKED: Raise PolicyViolation exception
   → If WARNING: Log and continue
   ↓
4. Instrumentation Engine Creates Span
   → trace_id, span_id generated
   → Start timestamp recorded
   ↓
5. Context Propagator Adds Attributes
   → team, project, environment
   ↓
6. Provider Adapter Wraps API Call
   → Original client method invoked
   → Request sent to AI provider API
   ↓
7. AI Provider Response Received
   ↓
8. Cost Calculator Computes
   → tokens × pricing = cost
   ↓
9. Evaluation Engine Records
   → Latency, tokens, quality metrics
   ↓
10. Instrumentation Engine Ends Span
    → End timestamp recorded
    → Duration calculated
    ↓
11. Enrichment Processor Adds Metadata
    → Cost, latency, derived attributes
    ↓
12. Filter Processor Applies Rules
    → PII removal, sampling
    ↓
13. Transform Processor Normalizes
    → Standardized attribute names
    ↓
14. Aggregation Processor Batches
    → Collect 100 spans or 10 seconds
    ↓
15. Exporter Factory Routes
    → Send to configured destinations
    ↓
16. OTLP/Datadog Export
    → HTTP/gRPC to observability platform
    ↓
17. OpenTelemetry Batch Processor
    → Async export, non-blocking
    ↓
18. Observability Platform Ingests
    → Datadog, Grafana, Honeycomb
    ↓
19. Dashboards Update
    → Real-time cost/usage visible
    ↓
20. Developer Receives Response
    → Original function returns normally
```

**Critical Path**: Steps 1-7, 8, 20 (application execution)
**Async Path**: Steps 11-19 (telemetry export, non-blocking)

## Performance Characteristics

### Latency Overhead

- **Decorator overhead**: <1ms per call
- **Span creation**: <0.5ms
- **Cost calculation**: <0.1ms
- **Policy evaluation**: <0.5ms
- **Total sync overhead**: ~2ms per AI call

**Percentage impact**:
- On 1000ms AI call: 0.2% overhead
- On 100ms AI call: 2% overhead
- On 10ms call (embeddings): 20% overhead

**Mitigation**: Async export ensures telemetry doesn't block application

### Memory Footprint

- **SDK base**: ~10MB
- **Per span**: ~2KB
- **Batch buffer** (100 spans): ~200KB
- **Provider adapters**: ~5MB total
- **Total**: ~15-20MB per application

### Throughput

- **Max spans/second**: 10,000+ (with batching)
- **Export batch size**: Configurable (default 100)
- **Export interval**: Configurable (default 10s)
- **Sampling**: Reduces volume proportionally

## Security Considerations

### PII Protection

**Default Behavior**:
- Prompt content: **NOT exported**
- Response content: **NOT exported**
- Only metadata exported (tokens, cost, model)

**Opt-in Content Logging**:
```python
GenOps.init(
    export_prompts=True,  # Explicitly enable
    pii_filter=PII_FILTER_AGGRESSIVE  # Redact emails, phones
)
```

### API Key Management

**Best Practices**:
- Environment variables (recommended)
- Secret managers (AWS Secrets Manager, Vault)
- Never in code or YAML files

**Validation**:
```python
# GenOps validates key format, not actual auth
# Prevents accidental exposure of invalid keys
```

### Network Security

- **TLS encryption**: All exports use HTTPS/gRPC with TLS
- **Certificate validation**: Enforced by default
- **Proxy support**: HTTP_PROXY environment variable

### RBAC Integration

**Team-based Access**:
```python
# Telemetry tagged with team attribute
# Observability platform enforces RBAC
# Team A cannot see Team B's telemetry
```

## Extensibility Points

### Custom Providers

```python
from genops.providers.base import ProviderAdapter

class MyCustomProvider(ProviderAdapter):
    def instrument(self, client):
        # Wrap client methods
        pass

    def calculate_cost(self, response):
        # Custom pricing logic
        pass

# Register with GenOps
GenOps.register_provider("mycustom", MyCustomProvider)
```

### Custom Policies

```python
from genops.core.policy import Policy

class CustomPolicy(Policy):
    def evaluate(self, context: Dict) -> PolicyDecision:
        # Custom logic
        if context["model"] == "gpt-4" and context["env"] == "dev":
            return PolicyDecision.BLOCKED
        return PolicyDecision.ALLOWED

GenOps.register_policy(CustomPolicy(name="dev_restrictions"))
```

### Custom Exporters

```python
from genops.exporters.base import Exporter

class CustomExporter(Exporter):
    def export(self, spans: List[Span]) -> ExportResult:
        # Send to custom destination
        requests.post("https://my-platform.com/api", json=spans)
        return ExportResult.SUCCESS

GenOps.add_exporter(CustomExporter())
```

### Custom Processors

```python
from genops.processors.base import Processor

class CustomProcessor(Processor):
    def process(self, span: Span) -> Span:
        # Custom transformation
        span.set_attribute("custom.field", compute_value())
        return span

GenOps.add_processor(CustomProcessor())
```

## Testing Strategies

### Unit Testing

```python
from genops.testing import MockExporter

def test_cost_calculation():
    exporter = MockExporter()
    GenOps.init(exporters=[exporter])

    # Make AI call
    response = openai.chat.completions.create(...)

    # Assert telemetry
    assert exporter.spans[0].attributes["genops.cost.total"] == 0.002
```

### Integration Testing

```bash
# Test full pipeline
genops test --provider openai --exporter datadog
# Validates: API call → instrumentation → export → platform ingestion
```

### Performance Testing

```python
# Benchmark overhead
import timeit

def without_genops():
    openai.chat.completions.create(...)

def with_genops():
    GenOps.init()
    openai.chat.completions.create(...)

overhead = timeit.timeit(with_genops, number=100) - timeit.timeit(without_genops, number=100)
print(f"Overhead: {overhead}ms per call")
```

## Conclusion

GenOps-AI's internal architecture demonstrates:

1. **Modularity**: Clear separation of concerns (config, core, providers, processors, exporters)
2. **Extensibility**: Plugin architecture for custom providers, policies, exporters
3. **Performance**: Async export, batching, sampling for minimal overhead
4. **Standards**: Built on OpenTelemetry for long-term compatibility
5. **Developer Experience**: Zero-code auto-instrumentation AND fine-grained control

The component design enables sophisticated governance while maintaining simplicity for end users.
