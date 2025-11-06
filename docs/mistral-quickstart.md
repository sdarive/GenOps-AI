# Mistral AI Integration - 5-Minute Quickstart

**🎯 Get GenOps tracking for Mistral AI models in 5 minutes**

This guide gets you from zero to tracking Mistral costs and performance with GenOps in under 5 minutes, featuring European AI provider benefits with GDPR compliance and competitive pricing.

---

## 🚀 Prerequisites (30 seconds)

**Before you start, make sure you have:**

1. **Mistral API key**
   ```bash
   # Get your API key from https://console.mistral.ai/
   export MISTRAL_API_KEY="your-mistral-api-key-here"
   ```

2. **Install Mistral client** (if not already installed)
   ```bash
   pip install mistralai
   ```

3. **Verify Mistral access**
   ```bash
   python -c "import mistralai; print('Mistral client ready')"
   ```

---

## ⚡ Quick Setup (2 minutes)

### Step 1: Install GenOps (30 seconds)
```bash
pip install genops-ai
```

### Step 2: Verify Setup (30 seconds)
Run this validation script to check everything is working:

```python
from genops.providers.mistral_validation import validate_setup, print_validation_result

# Check your Mistral setup
result = validate_setup()
print_validation_result(result)
```

You should see: ✅ **Overall Status: PASSED**

### Step 3: Test Basic Tracking (60 seconds)
Create this minimal test file:

```python
# test_mistral_genops.py
from genops.providers.mistral import instrument_mistral

# Enable GenOps tracking for Mistral (zero code changes needed!)
adapter = instrument_mistral(team="ai-team", project="quickstart-test")

print("🚀 Testing Mistral with GenOps tracking...")

# Generate text (costs and performance automatically tracked)
response = adapter.chat(
    message="What is the capital of France?",
    model="mistral-small-latest"
)

print(f"📝 Response: {response.content[:100]}...")
print(f"💰 Cost: ${response.usage.total_cost:.6f}")
print(f"🇪🇺 European AI: GDPR compliant, competitive pricing")
print("✅ SUCCESS! GenOps is now tracking your Mistral usage")
```

**Run it:**
```bash
python test_mistral_genops.py
```

**Expected output:**
```
🚀 Testing Mistral with GenOps tracking...
📝 Response: The capital of France is Paris. Paris is located in the north-central part of France...
💰 Cost: $0.000075
🇪🇺 European AI: GDPR compliant, competitive pricing
✅ SUCCESS! GenOps is now tracking your Mistral usage
```

---

## 🎯 What Just Happened?

**GenOps automatically tracked:**
- ✅ **Token-based costs** (input/output tokens with precise Mistral pricing)
- ✅ **Operation performance** (latency, tokens per second)
- ✅ **Team attribution** (costs attributed to "ai-team" and "quickstart-test")
- ✅ **European AI benefits** (GDPR compliance, cost competitiveness)
- ✅ **Model efficiency** (cost per operation, tokens per dollar)

**All with zero changes to your Mistral workflow!**

---

## 📊 See Your Data (1 minute)

### Option 1: Get Usage Summary
```python
from genops.providers.mistral import instrument_mistral

adapter = instrument_mistral(team="analytics-team")

# Run some operations first...
response1 = adapter.chat(message="Hello", model="mistral-small-latest")
response2 = adapter.embed(texts=["test document"], model="mistral-embed")

# Get comprehensive usage summary
summary = adapter.get_usage_summary()
print(f"💰 Total Cost: ${summary['total_cost']:.6f}")
print(f"🔢 Operations: {summary['total_operations']}")
print(f"⚡ Avg Cost/Op: ${summary['average_cost_per_operation']:.6f}")
print(f"🇪🇺 European AI advantages: GDPR + competitive pricing")
```

### Option 2: Multi-Operation Tracking
```python
from genops.providers.mistral import instrument_mistral

adapter = instrument_mistral(team="research-team", project="european-ai")

# Text generation with different models
chat_response = adapter.chat(
    message="Explain machine learning",
    model="mistral-large-2407"  # Premium model for complex tasks
)

# Cost-effective generation
simple_response = adapter.chat(
    message="What is 2+2?",
    model="mistral-tiny-2312"  # Ultra-low cost for simple tasks
)

# Text embedding
embed_response = adapter.embed(
    texts=["machine learning", "artificial intelligence", "European AI"],
    model="mistral-embed"
)

print(f"💬 Large model cost: ${chat_response.usage.total_cost:.6f}")
print(f"🔢 Tiny model cost: ${simple_response.usage.total_cost:.6f}")
print(f"📊 Embedding cost: ${embed_response.usage.total_cost:.6f}")
print(f"🇪🇺 Total European AI cost: ${chat_response.usage.total_cost + simple_response.usage.total_cost + embed_response.usage.total_cost:.6f}")
```

---

## 🏗️ Next Steps (Your Choice!)

**✅ You now have GenOps tracking all your Mistral operations!**

**Choose your next adventure:**

### 🎯 **30-Second Next Step: Try Different Models**
```python
# Compare costs across Mistral models (European AI efficiency)
from genops.providers.mistral import instrument_mistral

adapter = instrument_mistral(team="research", project="model-comparison")

models = [
    "mistral-tiny-2312",      # Ultra-low cost
    "mistral-small-latest",   # Cost-effective
    "mistral-medium-latest",  # Balanced performance
    "mistral-large-2407"      # Premium capabilities
]
prompt = "Explain quantum computing in one paragraph"

for model in models:
    response = adapter.chat(message=prompt, model=model)
    print(f"✅ {model}: ${response.usage.total_cost:.6f} ({response.usage.total_tokens} tokens)")
    
print("🇪🇺 European AI: Choose the right model for optimal cost-performance balance")
```

### 🚀 **5-Minute Next Step: European AI Advantages**
```python
# Explore European AI provider benefits
from genops.providers.mistral import instrument_mistral

adapter = instrument_mistral(team="compliance", project="eu-ai-benefits")

# GDPR-compliant text processing
gdpr_response = adapter.chat(
    message="Process this customer data according to GDPR requirements: [customer info]",
    model="mistral-small-latest"
)

# Cost-competitive analysis
analysis_response = adapter.chat(
    message="Compare European vs US AI regulations",
    model="mistral-medium-latest"
)

print("🇪🇺 **European AI Advantages:**")
print(f"   💰 Cost: ${gdpr_response.usage.total_cost + analysis_response.usage.total_cost:.6f}")
print("   ✅ GDPR compliant by default")
print("   🛡️ EU data residency")
print("   💸 Competitive pricing vs US providers")
print("   📊 No cross-border data transfer costs")
```

### 📚 **15-Minute Next Step: Complete Integration**
- **[Complete Mistral Integration Guide](../docs/integrations/mistral.md)** - Full reference documentation
- **[All Mistral Examples](../examples/mistral/)** - Progressive complexity tutorials
- **[European AI Compliance Guide](../docs/european-ai-compliance.md)** - GDPR and regulatory benefits

---

## 🆘 Troubleshooting

**Getting errors? Here are quick fixes:**

### ❌ "Invalid API key" or "Unauthorized"
```bash
# Make sure your API key is set correctly
echo $MISTRAL_API_KEY
# Should show your key (not empty)

# Or set it in Python
import os
os.environ["MISTRAL_API_KEY"] = "your-api-key-here"

# Verify key format - Mistral keys are different from OpenAI
# Get yours from: https://console.mistral.ai/
```

### ❌ "No module named 'mistralai'"
```bash
# Install Mistral Python client
pip install mistralai

# Verify installation
python -c "import mistralai; print('✅ Mistral installed')"
```

### ❌ "Model not found" or "Model not available"
```python
# Check available models for your account
from mistralai import Mistral
import os

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

# Try a basic model that should be available
try:
    response = client.chat.complete(
        model="mistral-tiny-2312",  # Cheapest model
        messages=[{"role": "user", "content": "test"}],
        max_tokens=1
    )
    print("✅ Mistral API working")
except Exception as e:
    print(f"❌ API Error: {e}")
```

### ❌ "Import error for genops"
```bash
# Reinstall GenOps
pip install --upgrade genops-ai
```

**Still stuck?** Run the diagnostic:
```python
from genops.providers.mistral_validation import validate_setup, print_validation_result
result = validate_setup()
print_validation_result(result, detailed=True)
```

---

## 💡 Key Differences from Other AI Providers

**Mistral tracking is optimized for European AI advantages:**

| Aspect | OpenAI/Anthropic (US) | Mistral (Europe) |
|--------|----------------------|------------------|
| **Data Residency** | US-based | EU-based (GDPR compliant) |
| **Cost Model** | Premium pricing | Competitive, cost-efficient |
| **Compliance** | Complex cross-border | Native GDPR compliance |
| **Specialization** | General purpose | European AI, multilingual |

**That's why GenOps Mistral integration focuses on:**
- 🇪🇺 **European AI advantages** (GDPR compliance, EU data residency)
- 💰 **Cost competitiveness** (20-60% savings vs US providers for similar performance)
- 🛡️ **Regulatory compliance** (native GDPR support without complexity)
- 📊 **Comprehensive cost attribution** with European data sovereignty benefits

---

## 🎉 Success!

**🎯 In 5 minutes, you've accomplished:**
- ✅ Set up GenOps tracking for Mistral AI operations
- ✅ Automatically tracked costs across different Mistral models
- ✅ Attributed costs to teams and projects
- ✅ Leveraged European AI provider advantages (GDPR + cost efficiency)
- ✅ Got insights into model performance and cost optimization

**Your Mistral AI operations now have enterprise-grade governance with European AI benefits!**

**🚀 Ready for more advanced features?** Check out:
- **[Multi-Model Examples](../examples/mistral/)**
- **[European AI Compliance Strategies](../docs/european-ai-compliance.md)**
- **[Complete Integration Guide](../docs/integrations/mistral.md)**

---

**Questions? Issues?** 
- 📝 [Create an issue](https://github.com/KoshiHQ/GenOps-AI/issues)
- 💬 [Join discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)
- 🇪🇺 [European AI Community](https://github.com/KoshiHQ/GenOps-AI/discussions/categories/european-ai)