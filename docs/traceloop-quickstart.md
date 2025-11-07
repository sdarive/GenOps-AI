# Traceloop + OpenLLMetry Quickstart Guide

**🎯 Add enterprise governance to your OpenLLMetry LLM observability in 5 minutes**

This quickstart gets you from zero to enhanced LLM observability with governance in exactly 5 minutes. OpenLLMetry provides the open-source foundation, with optional Traceloop commercial platform features.

---

## ⚡ 5-Minute Quick Start

### Step 1: Install (30 seconds)

```bash
pip install genops[traceloop]
```

This installs OpenLLMetry (open-source), Traceloop SDK (commercial platform), and GenOps governance enhancements.

### Step 2: Configure (30 seconds)

```bash
# Required: AI provider API key
export OPENAI_API_KEY="your-openai-api-key"

# Optional: Traceloop commercial platform
export TRACELOOP_API_KEY="your-traceloop-api-key"  # From app.traceloop.com
```

### Step 3: Validate Setup (30 seconds)

```bash
cd examples/traceloop
python setup_validation.py
```

**Expected output:** ✅ **Overall Status: PASSED**

### Step 4: Zero-Code Enhancement (30 seconds)

Add **one line** to your existing code:

```python
from genops.providers.traceloop import auto_instrument

# Enable governance for ALL your OpenLLMetry operations
auto_instrument(team="your-team", project="your-project")

# Your existing OpenLLMetry code now includes cost attribution and governance!
import openai
client = openai.OpenAI()

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello world!"}]
)
# ✅ Automatically tracked with team attribution and cost intelligence
```

### Step 5: See Results (2.5 minutes)

```bash
python basic_tracking.py
```

**You'll immediately see:**
- ✅ Enhanced OpenLLMetry traces with governance attributes
- 💰 Automatic cost attribution to your team and project
- 🛡️ Policy enforcement and budget monitoring
- 📊 Business intelligence integrated with observability

---

## 🎉 Success! You're Done!

**In 5 minutes you've added enterprise governance to your LLM operations.**

### What You Just Accomplished:
- Enhanced all OpenLLMetry operations with cost intelligence
- Added automatic team and project attribution
- Enabled policy enforcement and budget monitoring
- Maintained 100% compatibility with existing code

### Your Enhanced Observability Stack:
- **OpenLLMetry**: Open-source LLM observability foundation
- **GenOps**: Governance, cost intelligence, and policy enforcement
- **Traceloop** (optional): Commercial platform with advanced insights

---

## 🚀 Next Steps (Optional)

### Immediate Actions:
```bash
# Try zero-code enhancement on existing applications
python auto_instrumentation.py

# Explore commercial platform features (requires TRACELOOP_API_KEY)
python traceloop_platform.py
```

### Production Deployment:
```bash
# Enterprise patterns and high-availability
python production_patterns.py

# Advanced multi-provider governance
python advanced_observability.py
```

### Get Help:
- 📚 **Complete Guide**: [examples/traceloop/README.md](../examples/traceloop/README.md)
- 🛠️ **Integration Guide**: [docs/integrations/traceloop.md](integrations/traceloop.md)
- 💬 **Issues**: [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)

---

## 🔧 Common Issues

**❌ "OpenLLMetry not found"**
```bash
pip install openllmetry
```

**❌ "Validation failed"**
```bash
# Check your API key
echo $OPENAI_API_KEY  # Should be set

# Run validation with details
python setup_validation.py
```

**❌ "No cost attribution visible"**
- Ensure you called `auto_instrument()` before your OpenLLMetry operations
- Check that your observability backend supports OpenTelemetry attributes
- Verify governance attributes with: `python basic_tracking.py`

---

**Ready for production? You now have enterprise-grade LLM governance in 5 minutes! 🚀**