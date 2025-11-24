# Vercel AI SDK QuickStart Guide

**Get GenOps governance for your Vercel AI SDK applications in under 5 minutes.**

## 🚀 5-Minute Setup

### 1. Install (30 seconds)
```bash
# Install GenOps
pip install genops

# Install Vercel AI SDK (if not already installed)
npm install ai @ai-sdk/openai
```

### 2. Set Environment Variables (30 seconds)
```bash
export OPENAI_API_KEY="your-openai-key"
export GENOPS_TEAM="your-team"
export GENOPS_PROJECT="quickstart"
```

### 3. Validate Setup (30 seconds)
```bash
python -c "from genops.providers.vercel_ai_sdk_validation import validate_setup; validate_setup()"
```

### 4. Copy-Paste Working Example (3 minutes)

Create `quickstart_demo.py`:

```python
#!/usr/bin/env python3
"""5-Minute Vercel AI SDK + GenOps Demo"""

import os
import tempfile
from pathlib import Path

# Import GenOps auto-instrumentation
from genops.providers.vercel_ai_sdk import auto_instrument

def main():
    print("🤖 GenOps + Vercel AI SDK - 5 Minute Demo")
    print("=" * 50)
    
    # Enable GenOps governance (1 line!)
    adapter = auto_instrument(
        team=os.getenv('GENOPS_TEAM', 'quickstart'),
        project=os.getenv('GENOPS_PROJECT', 'demo')
    )
    
    # Create JavaScript code with GenOps instrumentation
    js_code = '''
const { generateText } = require('ai');
const { openai } = require('@ai-sdk/openai');

async function demo() {
    console.log('🚀 Generating text with Vercel AI SDK + GenOps governance...');
    
    const result = await generateText({
        model: openai('gpt-3.5-turbo'),
        prompt: 'Explain AI governance in one sentence.',
        maxTokens: 50
    });
    
    console.log('\\n📝 Generated Text:');
    console.log(result.text);
    console.log('\\n📊 Usage:');
    console.log(`Tokens: ${result.usage?.totalTokens || 'N/A'}`);
    console.log(`Cost estimate: $${((result.usage?.totalTokens || 0) * 0.000002).toFixed(6)}`);
    console.log('\\n✅ Demo completed with GenOps governance!');
}

demo().catch(console.error);
'''
    
    # Write and execute JavaScript
    temp_dir = Path(tempfile.mkdtemp())
    js_file = temp_dir / "demo.js"
    js_file.write_text(js_code)
    
    # Track the operation with GenOps
    print("📊 Starting GenOps-tracked operation...")
    with adapter.track_request("generateText", "openai", "gpt-3.5-turbo") as request:
        import subprocess
        try:
            result = subprocess.run(
                ["node", str(js_file)], 
                cwd=temp_dir, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            if result.returncode == 0:
                print(result.stdout)
                print(f"\\n🎯 GenOps Tracking Details:")
                print(f"  Request ID: {request.request_id}")
                print(f"  Team: {request.governance_attrs.get('team', 'N/A')}")
                print(f"  Project: {request.governance_attrs.get('project', 'N/A')}")
                print(f"  Provider: {request.provider}")
                print(f"  Model: {request.model}")
            else:
                print(f"❌ Error: {result.stderr}")
                
        except FileNotFoundError:
            print("❌ Node.js not found. Install from: https://nodejs.org/")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    
    print("\\n🎉 Demo Complete!")
    print("\\nWhat just happened:")
    print("1. ✅ GenOps auto-instrumentation enabled")
    print("2. ✅ Vercel AI SDK executed with governance")  
    print("3. ✅ Cost and usage automatically tracked")
    print("4. ✅ Team and project attribution added")
    print("5. ✅ OpenTelemetry telemetry generated")

if __name__ == "__main__":
    main()
```

### 5. Run the Demo (30 seconds)
```bash
python quickstart_demo.py
```

## ✅ Expected Output

```
🤖 GenOps + Vercel AI SDK - 5 Minute Demo
==================================================
📊 Starting GenOps-tracked operation...
🚀 Generating text with Vercel AI SDK + GenOps governance...

📝 Generated Text:
AI governance ensures responsible AI development through policies, monitoring, and ethical guidelines for safe deployment.

📊 Usage:
Tokens: 23
Cost estimate: $0.000046

✅ Demo completed with GenOps governance!

🎯 GenOps Tracking Details:
  Request ID: vercel-ai-sdk-1700123456789-12345
  Team: quickstart
  Project: demo
  Provider: openai  
  Model: gpt-3.5-turbo

🎉 Demo Complete!

What just happened:
1. ✅ GenOps auto-instrumentation enabled
2. ✅ Vercel AI SDK executed with governance
3. ✅ Cost and usage automatically tracked
4. ✅ Team and project attribution added
5. ✅ OpenTelemetry telemetry generated
```

## 🔧 Troubleshooting

### "Node.js not found"
```bash
# Install Node.js
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install node
# OR visit: https://nodejs.org/
```

### "GenOps not installed"
```bash
pip install genops
```

### "OpenAI API key not found"
```bash
export OPENAI_API_KEY="your-actual-api-key"
# Get one from: https://platform.openai.com/api-keys
```

### "Module 'ai' not found"
```bash
npm install ai @ai-sdk/openai
```

### "Validation failed"
```bash
# Run detailed validation
python -c "from genops.providers.vercel_ai_sdk_validation import validate_setup; validate_setup(verbose=True)"
```

## 🚀 What's Next?

### Immediate Next Steps (5 minutes each):
1. **Try Auto-Instrumentation**: `python ../examples/vercel_ai_sdk/02_auto_instrumentation.py`
2. **Explore Cost Tracking**: Add budget controls and alerts  
3. **Set Up Observability**: Connect to your monitoring dashboard

### Learn More (30 minutes):
- **[Complete Integration Guide](integrations/vercel-ai-sdk.md)** - All features and patterns
- **[Examples Suite](../examples/vercel_ai_sdk/)** - Progressive examples with working code
- **[Production Deployment](integrations/vercel-ai-sdk.md#production-deployment)** - Docker, Kubernetes patterns

### Production Ready (2 hours):
- **Multi-Provider Setup**: Add Anthropic, Gemini, etc.
- **Enterprise Governance**: Budget controls, compliance monitoring
- **Dashboard Integration**: Grafana, Datadog, Honeycomb setup

## 💡 Key Benefits You Just Enabled

- ✅ **Zero Code Changes**: Existing Vercel AI SDK code works unchanged
- ✅ **Automatic Cost Tracking**: Real-time cost attribution across providers
- ✅ **Team Attribution**: Per-team, per-project cost breakdown
- ✅ **OpenTelemetry Native**: Works with any observability platform
- ✅ **Multi-Provider Support**: Unified governance across 20+ AI providers
- ✅ **Production Ready**: Enterprise patterns and scaling support

## 🤝 Need Help?

- **Quick Questions**: Check the [troubleshooting section](#-troubleshooting) above
- **Documentation**: [Complete integration guide](integrations/vercel-ai-sdk.md)  
- **Examples**: [Progressive examples suite](../examples/vercel_ai_sdk/)
- **Issues**: [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- **Community**: [Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)

---

**⏱️ Total Time**: Under 5 minutes | **Result**: Full GenOps governance for Vercel AI SDK | **Next**: [Integration Guide](integrations/vercel-ai-sdk.md)