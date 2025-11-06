# Mistral AI Integration Guide

**Complete reference for integrating GenOps AI governance with Mistral AI's European AI platform**

This guide provides comprehensive documentation for all GenOps Mistral features, from basic cost tracking to advanced European AI optimization for enterprise GDPR-compliant workloads.

## Overview

GenOps provides complete governance for Mistral AI deployments including:

- **🇪🇺 European AI Provider Benefits** - Native GDPR compliance with EU data residency
- **💰 Cost-Competitive Pricing** - 20-60% savings vs US providers with transparent pricing
- **🔄 Multi-Model Tracking** - Unified cost tracking across chat, embedding, and specialized models
- **🎯 Enterprise Optimization** - Cost intelligence for European AI workloads and compliance requirements
- **🏷️ Team Attribution** - Attribute costs to teams, projects, and customers with GDPR compliance
- **⚡ Advanced Analytics** - Performance insights and recommendations for cost optimization
- **🛡️ Compliance Controls** - GDPR-native governance with audit trails and data sovereignty
- **📊 OpenTelemetry Integration** - Export to your existing European observability stack

## Quick Start

> **🚀 New to GenOps + Mistral?** Start with the [5-Minute Quickstart Guide](../mistral-quickstart.md) for an instant working example, then return here for comprehensive reference.

### Installation

```bash
# Install Mistral client
pip install mistralai

# Install GenOps
pip install genops-ai

# Set your API key
export MISTRAL_API_KEY="your-mistral-api-key"
```

### Basic Setup

```python
from genops.providers.mistral import instrument_mistral

# Enable comprehensive tracking for all Mistral operations
adapter = instrument_mistral(
    team="ai-team",
    project="european-ai"
)

# Your existing Mistral code now includes GenOps tracking
response = adapter.chat(
    message="What are the benefits of European AI?",
    model="mistral-small-latest"
)

# Multi-model workflow with cost optimization
large_response = adapter.chat(
    message="Analyze complex regulatory requirements for GDPR compliance",
    model="mistral-large-2407"  # Premium model for complex analysis
)

embeddings = adapter.embed(
    texts=["GDPR compliance", "European AI sovereignty"],
    model="mistral-embed"
)

# All operations automatically tracked with European AI governance
print(f"🇪🇺 European AI cost: ${response.usage.total_cost + large_response.usage.total_cost + embeddings.usage.total_cost:.6f}")
```

## Core Components

### 1. GenOpsMistralAdapter

The main adapter class for comprehensive Mistral instrumentation with European AI optimization.

```python
from genops.providers.mistral import GenOpsMistralAdapter

# Create adapter with advanced configuration
adapter = GenOpsMistralAdapter(
    api_key="your-api-key",  # Optional, uses MISTRAL_API_KEY env var
    
    # Cost tracking configuration
    cost_tracking_enabled=True,
    budget_limit=100.0,  # $100 budget limit
    cost_alert_threshold=0.8,  # 80% threshold for alerts
    
    # Governance defaults
    default_team="ml-engineering",
    default_project="european-ai-platform",
    default_environment="production",
    
    # Performance settings
    timeout=60.0,
    max_retries=3,
    enable_streaming=True
)
```

#### Chat Operations

```python
# Basic chat completion
response = adapter.chat(
    message="Explain GDPR requirements for AI systems",
    model="mistral-small-latest",
    team="compliance-team",
    project="gdpr-ai",
    customer_id="eu-customer-123"
)

# Advanced chat with system prompt
response = adapter.chat(
    message="Analyze this customer data",
    system_prompt="You are a GDPR-compliant AI assistant. Always consider data privacy.",
    model="mistral-medium-latest",
    temperature=0.3,
    max_tokens=500
)

# Cost-optimized chat for simple queries
simple_response = adapter.chat(
    message="What is 2+2?",
    model="mistral-tiny-2312",  # Ultra-low cost for simple tasks
    max_tokens=10
)

print(f"💰 Simple query cost: ${simple_response.usage.total_cost:.6f}")
print(f"🇪🇺 GDPR compliant: {simple_response.success}")
```

#### Text Generation

```python
# Text generation (alias for chat)
generated_text = adapter.generate(
    prompt="Write a GDPR-compliant privacy policy for AI applications:",
    model="mistral-large-2407",
    temperature=0.7,
    max_tokens=1000,
    team="legal-team",
    project="gdpr-compliance"
)

print(f"📝 Generated text: {generated_text.content[:200]}...")
print(f"💰 Generation cost: ${generated_text.usage.total_cost:.6f}")
```

#### Text Embeddings

```python
# Create embeddings for semantic search
embedding_response = adapter.embed(
    texts=[
        "European AI regulation compliance",
        "GDPR data processing requirements", 
        "EU data sovereignty principles",
        "Cross-border data transfer restrictions"
    ],
    model="mistral-embed",
    team="data-science",
    project="compliance-search"
)

print(f"📊 Embeddings created: {len(embedding_response.embeddings)}")
print(f"📏 Dimension: {embedding_response.embedding_dimension}")
print(f"💰 Embedding cost: ${embedding_response.usage.total_cost:.6f}")

# Use embeddings for semantic search
for i, embedding in enumerate(embedding_response.embeddings):
    print(f"  Vector {i+1}: {len(embedding)} dimensions")
```

### 2. European AI Cost Optimization

Mistral provides significant cost advantages for European organizations:

```python
from genops.providers.mistral_pricing import MistralPricingCalculator

# Compare European AI costs vs US providers
pricing_calc = MistralPricingCalculator()

# Analyze cost competitiveness
models_to_compare = [
    "mistral-tiny-2312",      # Ultra-low cost
    "mistral-small-latest",   # Cost-effective 
    "mistral-medium-latest",  # Balanced performance
    "mistral-large-2407"      # Premium capabilities
]

print("🇪🇺 European AI Cost Analysis:")
for model in models_to_compare:
    input_cost, output_cost, total_cost = pricing_calc.calculate_cost(
        model=model,
        operation="chat",
        input_tokens=1000,
        output_tokens=500
    )
    
    # Get model recommendations
    recommendations = pricing_calc.get_model_recommendations("GDPR compliance analysis")
    
    print(f"  {model}:")
    print(f"    Cost: ${total_cost:.6f} (1000 in, 500 out tokens)")
    print(f"    European benefits: GDPR compliance + EU data residency")

# Get optimization insights
insights = pricing_calc.get_optimization_insights(
    current_model="mistral-large-2407",
    operation="chat",
    input_tokens=500,
    output_tokens=200,
    use_case="GDPR compliance checking"
)

print(f"\n💡 European AI Optimization Insights:")
for insight in insights[:3]:  # Top 3 insights
    print(f"  • {insight.insight}")
    print(f"    Potential savings: ${insight.potential_savings:.6f}")
    print(f"    Action: {insight.recommended_action}")
```

### 3. Advanced Cost Analytics

```python
from genops.providers.mistral_cost_aggregator import MistralCostAggregator

# Create cost aggregator for European AI analytics
aggregator = MistralCostAggregator(
    retention_days=90,
    enable_real_time_alerts=True,
    cost_alert_threshold=50.0  # $50 daily alert threshold
)

# Set budgets for European teams
aggregator.set_budget("team", "eu-compliance", 200.0)  # $200/month
aggregator.set_budget("project", "gdpr-ai-platform", 500.0)  # $500/month
aggregator.set_budget("customer", "eu-enterprise-client", 1000.0)  # $1000/month

# Record operations (normally done automatically by adapter)
cost_breakdown = {
    "input_tokens": 800,
    "output_tokens": 400,
    "total_tokens": 1200,
    "input_cost": 0.0008,
    "output_cost": 0.0012,
    "total_cost": 0.002,
    "cost_per_token": 0.00000167
}

op_id = aggregator.record_operation(
    model="mistral-medium-latest",
    operation_type="chat",
    cost_breakdown=cost_breakdown,
    team="eu-compliance",
    project="gdpr-ai-platform",
    customer_id="eu-enterprise-client",
    environment="production"
)

# Get comprehensive cost summary
from genops.providers.mistral_cost_aggregator import TimeWindow

summary = aggregator.get_cost_summary(
    time_window=TimeWindow.DAY,
    team="eu-compliance"
)

print(f"🇪🇺 European AI Cost Summary:")
print(f"  Total cost: ${summary.total_cost:.6f}")
print(f"  Operations: {summary.total_operations}")
print(f"  Cost by model: {summary.cost_by_model}")
print(f"  GDPR compliance value: ${summary.gdpr_compliance_cost_savings:.6f}")
print(f"  EU data residency value: ${summary.eu_data_residency_value:.6f}")

# Get budget status
budget_status = aggregator.get_budget_status()
print(f"\n💰 Budget Status:")
for team, status in budget_status["teams"].items():
    print(f"  Team {team}: ${status['spent']:.2f}/${status['budget']:.2f} ({status['utilization_percent']:.1f}%)")
```

### 4. European AI Workflow Management

```python
from genops.providers.mistral import mistral_workflow_context

# GDPR-compliant document analysis workflow
with mistral_workflow_context(
    "gdpr_document_analysis",
    team="compliance-team",
    project="eu-regulatory-analysis",
    customer_id="european-bank",
    environment="production"
) as (ctx, workflow_id):
    
    print(f"🚀 Starting GDPR workflow: {workflow_id}")
    
    # Step 1: Analyze document for GDPR compliance
    compliance_analysis = ctx.chat(
        message="Analyze this document for GDPR compliance issues: [document content]",
        model="mistral-large-2407",  # Premium model for regulatory analysis
        temperature=0.2  # Low temperature for consistent compliance analysis
    )
    
    # Step 2: Generate compliance recommendations
    recommendations = ctx.chat(
        message=f"Based on this analysis: {compliance_analysis.content[:500]}, provide specific GDPR compliance recommendations",
        model="mistral-medium-latest",
        max_tokens=800
    )
    
    # Step 3: Create embeddings for compliance knowledge base
    compliance_embeddings = ctx.embed(
        texts=[
            compliance_analysis.content,
            recommendations.content,
            "GDPR Article 25 - Data protection by design",
            "GDPR Article 32 - Security of processing"
        ],
        model="mistral-embed"
    )
    
    # Workflow automatically tracks all costs and maintains GDPR compliance
    print(f"✅ GDPR workflow completed")
    print(f"💰 Total cost: ${ctx.get_usage_summary()['total_cost']:.6f}")
    print(f"🛡️ GDPR compliant: EU data residency maintained")
```

## European AI Advantages

### GDPR Compliance Benefits

Mistral AI provides native European AI capabilities with built-in GDPR compliance:

```python
# GDPR-compliant AI processing
def gdpr_compliant_analysis(data_to_process, data_subject_consent=True):
    if not data_subject_consent:
        return {"error": "GDPR consent required"}
    
    adapter = instrument_mistral(
        team="data-protection",
        project="gdpr-compliant-ai",
        environment="eu-production"
    )
    
    # Process data within EU jurisdiction
    response = adapter.chat(
        message=f"Analyze this data with GDPR compliance: {data_to_process}",
        model="mistral-medium-latest",
        system_prompt="Always maintain GDPR compliance. Do not store or log personal data."
    )
    
    return {
        "analysis": response.content,
        "gdpr_compliant": True,
        "data_residency": "EU",
        "cost": response.usage.total_cost,
        "jurisdiction": "European Union"
    }

# Example usage
result = gdpr_compliant_analysis(
    "Customer feedback about our European AI services",
    data_subject_consent=True
)

print(f"🇪🇺 GDPR Analysis Result:")
print(f"  Analysis: {result['analysis'][:200]}...")
print(f"  GDPR Compliant: {result['gdpr_compliant']}")
print(f"  Data Residency: {result['data_residency']}")
print(f"  Cost: ${result['cost']:.6f}")
```

### Cost Competitiveness Analysis

```python
def compare_european_vs_us_ai_costs():
    """Compare Mistral (European) vs US provider costs."""
    
    # Typical enterprise workload
    monthly_operations = 100000
    avg_input_tokens = 500
    avg_output_tokens = 300
    
    pricing_calc = MistralPricingCalculator()
    
    # Calculate Mistral costs
    mistral_cost = pricing_calc.estimate_monthly_cost(
        model="mistral-medium-latest",
        operations_per_day=monthly_operations // 30,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens
    )
    
    print("🇪🇺 European AI (Mistral) vs US Providers Cost Analysis:")
    print(f"  Monthly operations: {monthly_operations:,}")
    print(f"  Average tokens per operation: {avg_input_tokens + avg_output_tokens}")
    
    print(f"\n💰 Mistral AI (European):")
    print(f"  Monthly cost: ${mistral_cost['monthly_cost']:.2f}")
    print(f"  Cost per operation: ${mistral_cost['cost_per_operation']:.6f}")
    print(f"  Additional benefits:")
    print("    ✅ GDPR compliant by default")
    print("    ✅ EU data residency")
    print("    ✅ No cross-border data transfer costs")
    print("    ✅ Regulatory compliance simplified")
    
    # Estimate US provider costs (for comparison)
    estimated_us_cost = mistral_cost['monthly_cost'] * 1.4  # 40% higher estimate
    
    print(f"\n💸 Estimated US Provider:")
    print(f"  Monthly cost: ${estimated_us_cost:.2f}")
    print(f"  Additional compliance costs:")
    print("    ❌ GDPR compliance complexity: +$500-2000/month")
    print("    ❌ Cross-border data transfer setup: +$200-1000/month")
    print("    ❌ Legal/compliance overhead: +$1000-5000/month")
    
    total_savings = (estimated_us_cost - mistral_cost['monthly_cost']) + 1500  # Mid-range compliance costs
    print(f"\n🏆 European AI Advantage:")
    print(f"  Total monthly savings: ${total_savings:.2f}")
    print(f"  Annual savings: ${total_savings * 12:.2f}")
    print(f"  ROI on European AI: {(total_savings / mistral_cost['monthly_cost']) * 100:.1f}%")

# Run cost comparison
compare_european_vs_us_ai_costs()
```

## Production Deployment

### Enterprise Configuration

```python
# Enterprise-grade Mistral configuration
class EuropeanAIConfig:
    """Configuration for European AI deployment with GDPR compliance."""
    
    def __init__(self):
        self.adapter = GenOpsMistralAdapter(
            # European AI configuration
            cost_tracking_enabled=True,
            budget_limit=5000.0,  # $5K monthly limit
            cost_alert_threshold=0.8,
            
            # GDPR compliance defaults
            default_team="eu-ai-operations",
            default_project="gdpr-compliant-ai",
            default_environment="eu-production",
            
            # Performance for European latency
            timeout=90.0,  # Account for EU latency
            max_retries=3,
            enable_streaming=True
        )
        
        # Set up cost aggregation
        self.cost_aggregator = MistralCostAggregator(
            retention_days=365,  # Full year for compliance audits
            enable_real_time_alerts=True,
            cost_alert_threshold=500.0  # $500 daily alert
        )
        
        # Configure budgets for European teams
        self.setup_european_budgets()
    
    def setup_european_budgets(self):
        """Set up budgets for European organizational structure."""
        # Team budgets (monthly)
        european_teams = {
            "eu-compliance": 1000.0,
            "eu-customer-service": 2000.0,
            "eu-product-development": 1500.0,
            "eu-data-science": 1200.0
        }
        
        for team, budget in european_teams.items():
            self.cost_aggregator.set_budget("team", team, budget)
        
        # Customer budgets (monthly)
        eu_customers = {
            "german-bank": 3000.0,
            "french-retailer": 2500.0,
            "swedish-manufacturer": 1800.0
        }
        
        for customer, budget in eu_customers.items():
            self.cost_aggregator.set_budget("customer", customer, budget)
    
    def process_gdpr_request(self, request_type, customer_data, consent_verified=True):
        """Process GDPR data requests with full compliance."""
        if not consent_verified:
            return {"error": "GDPR consent not verified", "compliant": False}
        
        # Use European AI for GDPR-compliant processing
        response = self.adapter.chat(
            message=f"Process GDPR {request_type} request: {customer_data}",
            model="mistral-medium-latest",
            system_prompt="Ensure full GDPR compliance. Process data according to EU regulations.",
            team="eu-compliance",
            project="gdpr-data-requests",
            customer_id="gdpr-request-processing",
            temperature=0.1  # Consistent compliance processing
        )
        
        return {
            "response": response.content,
            "gdpr_compliant": True,
            "data_residency": "EU", 
            "processing_cost": response.usage.total_cost,
            "compliance_verified": True
        }

# Deploy European AI configuration
eu_ai = EuropeanAIConfig()

# Example GDPR request processing
gdpr_result = eu_ai.process_gdpr_request(
    request_type="data_portability",
    customer_data="Customer ID: EU-12345, requesting data export",
    consent_verified=True
)

print(f"🇪🇺 GDPR Request Processing:")
print(f"  Compliant: {gdpr_result['gdpr_compliant']}")
print(f"  Data residency: {gdpr_result['data_residency']}")
print(f"  Cost: ${gdpr_result['processing_cost']:.6f}")
```

### Monitoring and Observability

```python
# European AI monitoring with OpenTelemetry
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

def setup_european_ai_monitoring():
    """Set up monitoring for European AI operations."""
    
    # Configure OpenTelemetry for European data centers
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    # Export to European observability platform
    otlp_exporter = OTLPSpanExporter(
        endpoint="https://eu-observability.your-platform.com",
        headers={"x-region": "eu", "x-compliance": "gdpr"}
    )
    
    span_processor = BatchSpanProcessor(otlp_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    # Mistral operations will automatically create spans
    adapter = instrument_mistral(
        team="eu-monitoring",
        project="european-ai-observability"
    )
    
    # Test monitoring integration
    with tracer.start_as_current_span("european_ai_test") as span:
        span.set_attributes({
            "ai.provider": "mistral",
            "ai.region": "europe",
            "gdpr.compliant": True,
            "data.residency": "eu"
        })
        
        response = adapter.chat(
            message="Test European AI monitoring integration",
            model="mistral-small-latest"
        )
        
        span.set_attributes({
            "ai.cost": response.usage.total_cost,
            "ai.tokens": response.usage.total_tokens,
            "ai.model": "mistral-small-latest"
        })
    
    return adapter

# Set up monitoring
monitored_adapter = setup_european_ai_monitoring()
print("✅ European AI monitoring configured with GDPR compliance")
```

## API Reference

### GenOpsMistralAdapter Methods

#### `chat(message, model="mistral-small-latest", **kwargs)`

Generate chat completion with comprehensive cost tracking.

**Parameters:**
- `message` (str): User message content
- `model` (str): Mistral model to use (default: "mistral-small-latest")
- `system_prompt` (str, optional): System message for context
- `temperature` (float): Sampling temperature 0-1 (default: 0.7)
- `max_tokens` (int, optional): Maximum tokens to generate
- `stream` (bool): Whether to stream response (default: False)
- `team` (str, optional): Team attribution
- `project` (str, optional): Project attribution  
- `customer_id` (str, optional): Customer attribution
- `environment` (str): Environment (default: "development")

**Returns:** `MistralResponse` object with content, usage stats, and cost information

**Example:**
```python
response = adapter.chat(
    message="Explain GDPR Article 25",
    model="mistral-medium-latest",
    system_prompt="You are a GDPR compliance expert",
    temperature=0.3,
    max_tokens=500,
    team="compliance",
    project="gdpr-analysis"
)
```

#### `embed(texts, model="mistral-embed", **kwargs)`

Generate text embeddings with cost tracking.

**Parameters:**
- `texts` (Union[str, List[str]]): Text(s) to embed
- `model` (str): Embedding model (default: "mistral-embed")
- Governance parameters: `team`, `project`, `customer_id`, `environment`

**Returns:** `MistralResponse` object with embeddings and cost information

#### `generate(prompt, model="mistral-small-latest", **kwargs)`

Generate text completion (alias for chat with single message).

### Cost Analysis Methods

#### `get_usage_summary()`

Get comprehensive usage summary for current session.

**Returns:** Dictionary with cost, operations, and efficiency metrics

#### `reset_session_stats()`

Reset session-level statistics for new cost tracking period.

### European AI Utilities

#### `mistral_workflow_context(workflow_name, **governance_attrs)`

Context manager for European AI workflow cost tracking.

**Example:**
```python
with mistral_workflow_context("gdpr_analysis", team="compliance") as (ctx, workflow_id):
    # All operations automatically tracked with European governance
    analysis = ctx.chat("Analyze GDPR compliance", model="mistral-medium-latest")
```

## Model Selection Guide

### European AI Model Recommendations

| Use Case | Recommended Model | Cost/1M Tokens | GDPR Features |
|----------|------------------|----------------|---------------|
| **Simple Q&A** | `mistral-tiny-2312` | $0.25 | ✅ EU residency |
| **General Chat** | `mistral-small-latest` | $1-3 | ✅ GDPR compliant |
| **Content Generation** | `mistral-medium-latest` | $2.75-8.10 | ✅ EU processing |
| **Complex Analysis** | `mistral-large-2407` | $8-24 | ✅ Advanced compliance |
| **Code Generation** | `codestral-2405` | $3 | ✅ IP protection |
| **Embeddings** | `mistral-embed` | $0.10 | ✅ Semantic search |
| **Long Documents** | `mistral-nemo-2407` | $1 | ✅ 128K context |

### Cost Optimization Strategies

1. **Task-Based Model Selection**
   ```python
   def select_model_by_complexity(task_description):
       """Select optimal Mistral model based on task complexity."""
       complexity_keywords = {
           "simple": ["yes", "no", "basic", "simple"],
           "medium": ["explain", "analyze", "generate", "write"],
           "complex": ["research", "legal", "compliance", "detailed"]
       }
       
       task_lower = task_description.lower()
       
       if any(keyword in task_lower for keyword in complexity_keywords["simple"]):
           return "mistral-tiny-2312"  # Ultra-low cost
       elif any(keyword in task_lower for keyword in complexity_keywords["complex"]):
           return "mistral-large-2407"  # Premium capabilities
       else:
           return "mistral-small-latest"  # Cost-effective default
   
   # Usage
   optimal_model = select_model_by_complexity("Simple yes/no question")
   response = adapter.chat(message="Is Paris in France?", model=optimal_model)
   ```

2. **European Compliance Optimization**
   ```python
   def gdpr_optimized_processing(data_type, complexity="medium"):
       """Process data with GDPR optimization."""
       
       # Select model based on data sensitivity and complexity
       if data_type == "personal_data":
           # Use EU-resident model with enhanced privacy
           model = "mistral-medium-latest"
           temp = 0.1  # Low temperature for consistent compliance
       elif complexity == "high":
           model = "mistral-large-2407"
           temp = 0.3
       else:
           model = "mistral-small-latest" 
           temp = 0.7
       
       return {
           "model": model,
           "temperature": temp,
           "gdpr_optimized": True,
           "eu_residency": True
       }
   ```

## Troubleshooting

### Common Issues and Solutions

#### Authentication Issues

```python
# Test Mistral API connectivity
def test_mistral_connection():
    """Test Mistral API connection with error diagnosis."""
    import os
    
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return {
            "error": "API key not found",
            "solution": "Set MISTRAL_API_KEY environment variable",
            "get_key": "https://console.mistral.ai/"
        }
    
    try:
        from mistralai import Mistral
        client = Mistral(api_key=api_key)
        
        # Test with minimal cost
        response = client.chat.complete(
            model="mistral-tiny-2312",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        
        return {"status": "success", "connection": "working"}
        
    except Exception as e:
        error_msg = str(e).lower()
        
        if "unauthorized" in error_msg:
            return {
                "error": "Authentication failed",
                "solution": "Check API key is correct and active",
                "verify": "Visit https://console.mistral.ai/ to verify key"
            }
        elif "quota" in error_msg or "insufficient" in error_msg:
            return {
                "error": "Insufficient credits",
                "solution": "Add credits to your Mistral account",
                "billing": "https://console.mistral.ai/billing"
            }
        else:
            return {
                "error": f"Connection failed: {e}",
                "solution": "Check internet connection and Mistral service status"
            }

# Run connection test
connection_status = test_mistral_connection()
print(f"🔍 Connection Status: {connection_status}")
```

#### Cost Tracking Issues

```python
# Validate cost tracking setup
def validate_cost_tracking():
    """Validate GenOps cost tracking is working correctly."""
    try:
        from genops.providers.mistral_pricing import MistralPricingCalculator
        
        calc = MistralPricingCalculator()
        
        # Test cost calculation
        input_cost, output_cost, total_cost = calc.calculate_cost(
            model="mistral-small-latest",
            operation="chat",
            input_tokens=100,
            output_tokens=50
        )
        
        if total_cost > 0:
            return {
                "status": "working",
                "test_cost": total_cost,
                "pricing_available": True
            }
        else:
            return {
                "status": "issue",
                "error": "Cost calculation returned zero",
                "solution": "Check pricing calculator configuration"
            }
            
    except ImportError as e:
        return {
            "status": "error",
            "error": f"Import failed: {e}",
            "solution": "Reinstall genops-ai: pip install --upgrade genops-ai"
        }

# Validate cost tracking
cost_status = validate_cost_tracking()
print(f"💰 Cost Tracking: {cost_status}")
```

### Performance Optimization

```python
def optimize_mistral_performance():
    """Optimize Mistral performance for European latency."""
    
    config = {
        # European-optimized settings
        "timeout": 120.0,  # Account for EU latency
        "max_retries": 3,
        "enable_streaming": True,  # Better for long responses
        
        # Cost optimization
        "cost_tracking_enabled": True,
        "budget_limit": 1000.0,
        "cost_alert_threshold": 0.8,
        
        # Model selection optimization
        "default_models": {
            "simple": "mistral-tiny-2312",
            "standard": "mistral-small-latest", 
            "complex": "mistral-medium-latest",
            "premium": "mistral-large-2407"
        }
    }
    
    return config

# Apply optimizations
perf_config = optimize_mistral_performance()
optimized_adapter = GenOpsMistralAdapter(**perf_config)
```

## Migration Guide

### From OpenAI to Mistral

```python
def migrate_openai_to_mistral():
    """Migration helper from OpenAI to Mistral European AI."""
    
    migration_map = {
        # OpenAI -> Mistral model mapping
        "gpt-3.5-turbo": "mistral-small-latest",
        "gpt-4": "mistral-medium-latest", 
        "gpt-4-turbo": "mistral-large-2407",
        "text-embedding-ada-002": "mistral-embed"
    }
    
    def convert_openai_call(openai_params):
        """Convert OpenAI API call to Mistral."""
        
        # Map model
        openai_model = openai_params.get("model", "gpt-3.5-turbo")
        mistral_model = migration_map.get(openai_model, "mistral-small-latest")
        
        # Convert parameters
        mistral_params = {
            "model": mistral_model,
            "temperature": openai_params.get("temperature", 0.7),
            "max_tokens": openai_params.get("max_tokens"),
        }
        
        # Handle messages format
        if "messages" in openai_params:
            messages = openai_params["messages"]
            if len(messages) == 1:
                mistral_params["message"] = messages[0]["content"]
            else:
                # Handle system + user messages
                system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
                user_msg = next((m["content"] for m in messages if m["role"] == "user"), None)
                
                mistral_params["message"] = user_msg
                if system_msg:
                    mistral_params["system_prompt"] = system_msg
        
        return mistral_params
    
    # Example migration
    openai_request = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain GDPR compliance"}
        ],
        "temperature": 0.3,
        "max_tokens": 500
    }
    
    mistral_request = convert_openai_call(openai_request)
    
    print("🔄 OpenAI to Mistral Migration:")
    print(f"  OpenAI model: {openai_request['model']}")
    print(f"  Mistral model: {mistral_request['model']}")
    print(f"  European benefits: GDPR compliant + cost savings")
    
    return mistral_request

# Run migration
migrated_params = migrate_openai_to_mistral()
```

### Cost Comparison Tool

```python
def compare_migration_costs(monthly_operations=10000, avg_tokens=800):
    """Compare costs between providers for migration planning."""
    
    from genops.providers.mistral_pricing import MistralPricingCalculator
    
    calc = MistralPricingCalculator()
    
    # Calculate Mistral costs
    mistral_monthly = calc.estimate_monthly_cost(
        model="mistral-medium-latest",
        operations_per_day=monthly_operations // 30,
        avg_input_tokens=int(avg_tokens * 0.6),  # 60% input
        avg_output_tokens=int(avg_tokens * 0.4)   # 40% output
    )
    
    # Estimated OpenAI costs (for comparison)
    openai_monthly_estimate = mistral_monthly['monthly_cost'] * 1.8  # ~80% higher
    
    # Additional European benefits
    gdpr_compliance_savings = 1500  # Monthly compliance cost savings
    data_residency_value = 500      # Monthly data residency value
    
    print("💰 Migration Cost Analysis:")
    print(f"  Monthly operations: {monthly_operations:,}")
    print(f"  Average tokens per operation: {avg_tokens}")
    
    print(f"\n🇪🇺 Mistral AI (European):")
    print(f"  Direct costs: ${mistral_monthly['monthly_cost']:.2f}/month")
    print(f"  GDPR compliance savings: ${gdpr_compliance_savings:.2f}/month")
    print(f"  Data residency value: ${data_residency_value:.2f}/month") 
    print(f"  Total value: ${mistral_monthly['monthly_cost'] + gdpr_compliance_savings + data_residency_value:.2f}/month")
    
    print(f"\n🇺🇸 OpenAI (Estimated):")
    print(f"  Direct costs: ${openai_monthly_estimate:.2f}/month")
    print(f"  GDPR compliance costs: +${gdpr_compliance_savings:.2f}/month")
    print(f"  Cross-border transfer costs: +$200-1000/month")
    print(f"  Legal/compliance overhead: +$1000-3000/month")
    
    total_savings = (openai_monthly_estimate + gdpr_compliance_savings + 600 + 2000) - mistral_monthly['monthly_cost']
    
    print(f"\n🏆 Migration Benefits:")
    print(f"  Monthly savings: ${total_savings:.2f}")
    print(f"  Annual savings: ${total_savings * 12:.2f}")
    print(f"  ROI: {(total_savings / mistral_monthly['monthly_cost']) * 100:.1f}%")
    print(f"  Payback period: Immediate (compliance benefits)")
    
    return {
        "mistral_monthly": mistral_monthly['monthly_cost'],
        "estimated_savings": total_savings,
        "roi_percent": (total_savings / mistral_monthly['monthly_cost']) * 100
    }

# Run cost comparison
migration_analysis = compare_migration_costs()
```

---

## Support and Resources

### Documentation
- **[5-Minute Quickstart](../mistral-quickstart.md)** - Get started immediately
- **[European AI Examples](../../examples/mistral/)** - Progressive tutorials
- **[GDPR Compliance Guide](../european-ai-compliance.md)** - Regulatory best practices

### Community
- **[GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)** - Questions and community help
- **[European AI Community](https://github.com/KoshiHQ/GenOps-AI/discussions/categories/european-ai)** - Specific European AI discussions
- **[GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)** - Bug reports and feature requests

### Professional Services
- **GDPR Compliance Consulting** - Expert guidance for AI compliance
- **Migration Services** - Professional migration from US to European AI providers  
- **Enterprise Support** - Dedicated support for European organizations

---

**Ready to leverage European AI advantages with GenOps governance?**

🇪🇺 **Start with**: [5-Minute Quickstart](../mistral-quickstart.md)  
📊 **Explore**: [European AI Examples](../../examples/mistral/)  
🛡️ **Compliance**: [GDPR AI Guide](../european-ai-compliance.md)