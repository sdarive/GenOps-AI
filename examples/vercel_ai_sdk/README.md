# Vercel AI SDK + GenOps Examples

**🚀 Get GenOps governance for your Vercel AI SDK applications in 5 minutes.**

> **New to Vercel AI SDK?** It's a TypeScript toolkit for building AI apps with React, Next.js, Vue, Svelte & Node.js. Works with 20+ AI providers (OpenAI, Anthropic, Google, etc.). **GenOps adds cost tracking, team attribution, and governance** - with zero code changes!

## 🎯 Start Here (5 Minutes)

### 1. One-Command Setup
```bash
pip install genops && npm install ai @ai-sdk/openai
export OPENAI_API_KEY="your-key" GENOPS_TEAM="your-team"
```

### 2. Copy-Paste Demo
```bash
# Download and run immediately (if using from GitHub)
curl -O https://raw.githubusercontent.com/KoshiHQ/GenOps-AI/main/examples/vercel_ai_sdk/01_basic_text_generation.py
python 01_basic_text_generation.py

# Or if you have the repo locally:
python 01_basic_text_generation.py
```

### 3. See Immediate Results
```
✅ GenOps governance enabled
💰 Cost tracking: $0.000046 for 23 tokens
📊 Team attribution: your-team
🔍 Request ID: vercel-ai-sdk-1700123456789
```

**🎉 Success!** You now have full GenOps governance for Vercel AI SDK.

## 📚 Progressive Learning Path

### ⭐ **Beginner (5 minutes each)**
| Example | What You'll Learn | Time |
|---------|-------------------|------|
| **[01. Basic Text Generation](01_basic_text_generation.py)** | Core governance setup | 5 min |
| **[02. Auto-Instrumentation](02_auto_instrumentation.py)** | Zero-code integration | 5 min |

**Ready for more?** ⬇️

### ⭐⭐ **Intermediate (Coming Soon!)**
| Example | What You'll Learn | Status |
|---------|-------------------|--------|
| **03. Streaming Chat** | Real-time cost tracking | 🚧 Coming Soon |
| **04. Structured Data** | Object generation governance | 🚧 Coming Soon |

### ⭐⭐⭐ **Advanced (Coming Soon!)**
| Example | What You'll Learn | Status |
|---------|-------------------|--------|
| **05. Multi-Provider Routing** | Cost optimization across providers | 🚧 Coming Soon |
| **06. Agent Workflows** | Complex tool-calling governance | 🚧 Coming Soon |
| **07. Production Next.js** | Full application integration | 🚧 Coming Soon |
| **08. Enterprise Governance** | Complete enterprise deployment | 🚧 Coming Soon |

**Want these examples?** [Star the repo](https://github.com/KoshiHQ/GenOps-AI) and [open an issue](https://github.com/KoshiHQ/GenOps-AI/issues) requesting the specific examples you need!

## 📖 Complete Documentation

**For comprehensive information:**
- 📚 **[Complete Integration Guide](../../docs/integrations/vercel-ai-sdk.md)** - Production deployment, API reference, advanced patterns
- 🚀 **[5-Minute Quickstart](../../docs/vercel-ai-sdk-quickstart.md)** - Get started immediately
- 🛠️ **[Setup Validation](setup_validation.py)** - Diagnostic tool for troubleshooting

## 🔧 Quick Troubleshooting

**"Node.js not found"**
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install node
```

**"GenOps not installed"**
```bash
pip install genops
```

**"API key not found"**
```bash
export OPENAI_API_KEY="your-actual-key"
```

**"Still not working?"**
```bash
python setup_validation.py  # Comprehensive diagnostic
```

## Architecture Patterns

### Python Wrapper Pattern
Use GenOps Python adapter to wrap and instrument Vercel AI SDK calls:
```python
from genops.providers.vercel_ai_sdk import auto_instrument

adapter = auto_instrument(team="ai-team", project="chatbot")
with adapter.track_request("generateText", "openai", "gpt-4") as request:
    # Your Vercel AI SDK JavaScript code here
    pass
```

### WebSocket Bridge Pattern  
Real-time telemetry streaming between JavaScript and Python:
```python
adapter = GenOpsVercelAISDKAdapter(integration_mode="websocket")
# JavaScript client sends telemetry to Python via WebSocket
```

### Subprocess Integration
Execute instrumented Node.js scripts from Python:
```python
adapter = GenOpsVercelAISDKAdapter(integration_mode="subprocess")
result = adapter.execute_instrumented_script("generateText", options)
```

## JavaScript Integration

### Auto-Instrumentation Setup
```javascript
// Generated instrumentation code
const { generateText, streamText } = require('./genops-vercel-instrumentation');

// Your existing code works unchanged - governance added automatically!
const result = await generateText({
  model: 'gpt-4',
  prompt: 'Hello, world!'
});
```

### Manual Instrumentation
```javascript
const { track_generate_text } = require('genops-vercel-sdk');

await track_generate_text('openai', 'gpt-4', {
  team: 'ai-team',
  project: 'chatbot',
  prompt: 'Hello, world!'
});
```

## Cost Tracking Features

- **Multi-Provider Support**: Unified cost tracking across 20+ AI providers
- **Real-Time Monitoring**: Live cost updates during streaming operations
- **Budget Controls**: Automatic budget enforcement and alerting
- **Team Attribution**: Per-team, per-project, per-customer cost breakdown
- **Usage Analytics**: Detailed usage patterns and optimization insights

## Production Deployment

### Docker Integration
```dockerfile
FROM node:18-alpine
RUN npm install ai @ai-sdk/openai
COPY genops-vercel-instrumentation.js .
# Your application code
```

### Kubernetes Patterns
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vercel-ai-app
spec:
  template:
    spec:
      containers:
      - name: app
        env:
        - name: GENOPS_TEAM
          value: "production-team"
        - name: OTEL_EXPORTER_OTLP_ENDPOINT
          value: "http://jaeger:14268/api/traces"
```

## Troubleshooting

### Common Issues

**1. Node.js Not Found**
```bash
# Install Node.js
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install node
```

**2. Vercel AI SDK Not Installed**
```bash
npm install ai @ai-sdk/openai
```

**3. Missing API Keys**
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

**4. WebSocket Connection Failed**
```bash
# Check if port is available
netstat -an | grep 8080
# Try different port
export GENOPS_WEBSOCKET_PORT=8081
```

### Validation Tools

Run comprehensive setup validation:
```python
from genops.providers.vercel_ai_sdk_validation import validate_setup
result = validate_setup(verbose=True)
if not result.all_passed:
    print("Fix required issues and try again")
```

Quick health check:
```python
from genops.providers.vercel_ai_sdk_validation import quick_validation
if quick_validation():
    print("✅ Ready to go!")
else:
    print("❌ Setup issues detected")
```

## Integration Modes

### 1. Python Wrapper Mode (Recommended)
- **Best for**: Python-heavy applications
- **Setup**: Import and use Python adapter
- **Pros**: Full Python integration, easy debugging
- **Cons**: Requires subprocess for JavaScript execution

### 2. WebSocket Bridge Mode
- **Best for**: Real-time applications
- **Setup**: Start WebSocket server, connect JavaScript client
- **Pros**: Real-time telemetry, low latency
- **Cons**: More complex setup, requires WebSocket support

### 3. Subprocess Mode
- **Best for**: Batch processing
- **Setup**: Execute Node.js scripts from Python
- **Pros**: Simple integration, good for scripts
- **Cons**: Higher overhead, limited real-time features

## Performance Considerations

- **Telemetry Overhead**: <5ms per request
- **Memory Usage**: ~10MB for adapter
- **Network**: OTLP export in batches
- **Sampling**: Configurable for high-volume applications

## 🤝 Support & Next Steps

### **Need Help?**
- 🚀 **[5-Minute Quickstart](../../docs/vercel-ai-sdk-quickstart.md)** - Start here if you're new
- 📚 **[Complete Integration Guide](../../docs/integrations/vercel-ai-sdk.md)** - Comprehensive documentation
- 🔧 **[Setup Validation](setup_validation.py)** - Run diagnostic checks
- 🐛 **[GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)** - Report bugs and request features
- 💬 **[Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)** - Community help and tips

### **Ready for Production?**
- 🐳 **Docker & Kubernetes**: See [integration guide](../../docs/integrations/vercel-ai-sdk.md#production-deployment)
- 🏢 **Enterprise Deployment**: Full governance patterns and scaling
- 📊 **Monitoring Setup**: Grafana, Datadog, Honeycomb integration
- 🛡️ **Security & Compliance**: Enterprise governance templates

---

**⏰ Total Setup Time**: 5 minutes | **✨ Result**: Full GenOps governance for Vercel AI SDK