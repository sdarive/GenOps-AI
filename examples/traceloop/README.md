# Traceloop + OpenLLMetry LLM Observability + GenOps Governance Examples

**🎯 Add enterprise governance to your OpenLLMetry LLM observability in 5 minutes**

This directory contains comprehensive examples demonstrating how GenOps enhances OpenLLMetry with enterprise-grade governance, cost intelligence, and policy enforcement for production AI applications, with optional integration to the Traceloop commercial platform.

---

## 🤔 Why Do I Need This?

If you're building production LLM applications, you're likely facing these challenges:

❌ **Without GenOps Governance:**
- No visibility into LLM costs across teams and projects
- Manual budget tracking and cost attribution
- No policy enforcement or compliance validation
- Limited observability context for business decisions
- Difficult to optimize costs or prevent budget overruns

✅ **With GenOps + OpenLLMetry + Traceloop:**
- **Automatic cost attribution** to teams, projects, and customers
- **Real-time budget enforcement** with policy compliance
- **Enhanced observability** with business context in every trace
- **Cost optimization insights** and recommendations
- **Enterprise governance** for compliance and audit requirements
- **Optional commercial platform** for advanced insights and analytics

---

## 🧠 What is This Integration?

**GenOps + OpenLLMetry + Traceloop** = Complete LLM observability with enterprise governance

### 🏗️ The Stack
- **🔍 OpenLLMetry**: Open-source LLM observability framework (Apache 2.0, vendor-neutral)
- **🏢 Traceloop**: Commercial platform with advanced insights and team collaboration
- **🛡️ GenOps**: Governance layer adding cost intelligence and policy enforcement

### ✨ Key Benefits
- **🔍 Enhanced Observability**: Every trace includes governance context (team, project, customer)
- **💰 Cost Intelligence**: Precise cost tracking and attribution integrated with observability  
- **🛡️ Policy Compliance**: Real-time governance and budget enforcement
- **📊 Business Intelligence**: Cost optimization insights with team-based attribution
- **🚀 Enterprise Ready**: Production-grade governance for LLM observability at scale
- **🏭 Optional Commercial**: Upgrade to Traceloop platform for advanced features

---

## ⚡ Quick Value Assessment (2 minutes)

**Before diving in, let's see if this is right for your team:**

### ✅ Perfect For:
- **Engineering Teams** using or considering OpenLLMetry who need cost visibility and governance
- **FinOps Teams** requiring detailed LLM cost attribution and budget controls
- **Enterprise Organizations** needing compliance tracking and audit trails for AI operations
- **Multi-team Companies** where different teams use LLMs with shared budgets
- **Production AI Applications** requiring cost optimization and governance automation

### 🤔 Consider Alternatives If:
- You have simple, single-developer LLM projects with no cost concerns
- You only need basic cost tracking without detailed observability
- You don't plan to use OpenTelemetry-based observability practices

**📊 Team Size Guidelines:**
- **1-2 developers**: Start with Level 1 examples (basic governance with open-source OpenLLMetry)
- **3-10 developers**: Focus on Level 2 (advanced observability and evaluation)
- **10+ developers**: Implement Level 3 (enterprise governance and consider Traceloop platform)

---

## 💰 ROI & Business Value

### Small Teams (1-5 developers)
**Investment:** ~2 hours setup  
**Savings:** 20-40% LLM cost reduction through optimization  
**Value:** Clear cost visibility and basic governance

### Growing Teams (5-20 developers)  
**Investment:** ~1 day implementation  
**Savings:** 30-50% cost reduction + 50% faster debugging  
**Value:** Team attribution, budget controls, evaluation workflows

### Enterprise (20+ developers)
**Investment:** ~1 week enterprise deployment  
**Savings:** 40-60% cost reduction + compliance automation  
**Value:** Full governance automation, audit trails, enterprise observability

---

## 🚀 Getting Started (5 Minutes Total)

### Step 1: Install & Setup (2 minutes)

```bash
# Install GenOps with Traceloop + OpenLLMetry integration
pip install genops[traceloop]

# Set up your AI provider API key (choose one)
export OPENAI_API_KEY="your-openai-api-key"             # Recommended
export ANTHROPIC_API_KEY="your-anthropic-api-key"       # Alternative

# Optional: Traceloop commercial platform features
export TRACELOOP_API_KEY="your-traceloop-api-key"       # From app.traceloop.com
```

**Prerequisites:**
- **Python 3.8+** 
- **AI Provider Account**: [OpenAI Platform](https://platform.openai.com/api-keys) or [Anthropic Console](https://console.anthropic.com/)
- **Optional**: [Traceloop Platform Account](https://app.traceloop.com/) for commercial features

### Step 2: Validate Setup (30 seconds)

**🎯 Always run this first:**

```bash
cd examples/traceloop
python setup_validation.py
```

**Expected output:** ✅ **Overall Status: PASSED**

### Step 3: See Immediate Value (2.5 minutes)

```bash
# Zero-code governance integration
python auto_instrumentation.py  # 1 line of code adds governance to ALL operations

# Enhanced observability with cost attribution
python basic_tracking.py       # See governance in your traces
```

**🎉 Success!** You now have enterprise governance for your LLM operations.

---

## 🆘 Quick Troubleshooting

**❌ "ModuleNotFoundError: No module named 'openllmetry'"**
```bash
pip install openllmetry
# Or reinstall with: pip install genops[traceloop]
```

**❌ "No LLM provider API keys found"**
```bash
# Verify at least one provider is configured
echo $OPENAI_API_KEY       # Should be set if using OpenAI
echo $ANTHROPIC_API_KEY    # Should be set if using Anthropic
```

**❌ "Governance integration issues"**
```bash
# Enable detailed logging for diagnosis
export GENOPS_LOG_LEVEL=DEBUG
python basic_tracking.py
```

**Need more help?** See the [Advanced Troubleshooting](#-advanced-troubleshooting) section below.

---

## 📚 Learning Path Guide

### 🎯 Your Progressive Journey

**⏱️ Time Investment:** 4-6 hours (spread across days/weeks)  
**🚀 Immediate Value:** Visible in first 5 minutes  
**🏭 Production Ready:** After Level 2 completion  

---

### 🟢 Level 1: Getting Started (15 minutes total)
**🎯 Goal:** See immediate value and understand the integration  
**🏷️ Best For:** Initial evaluation, proof-of-concept, team demos

**🎓 What You'll Learn:**
- How to add governance to existing OpenLLMetry applications with zero code changes
- See cost attribution and team tracking in your observability platform  
- Understand the relationship between OpenLLMetry, Traceloop, and GenOps
- Experience enhanced traces with governance context

**📝 Examples to Run:**

1. **[setup_validation.py](setup_validation.py)** ⭐ *Always start here* (30 seconds)
   - Validates your complete setup with actionable diagnostics
   - Tests connectivity, API keys, and governance integration
   - Shows you exactly what's working and what needs attention

2. **[auto_instrumentation.py](auto_instrumentation.py)** (5 minutes)  
   - **Zero-code magic**: Add one line, get governance for ALL operations
   - Perfect if you already use OpenLLMetry patterns
   - Demonstrates compatibility with existing applications

3. **[basic_tracking.py](basic_tracking.py)** (5 minutes)
   - See governance attributes integrated with OpenLLMetry traces
   - Experience cost attribution and team tracking  
   - Learn manual instrumentation patterns for custom use cases

**✅ Level 1 Success Criteria:**
- [ ] Validation script shows ✅ **Overall Status: PASSED**
- [ ] You can see cost attribution in your observability dashboard
- [ ] You understand how GenOps enhances OpenLLMetry without replacing it
- [ ] Your existing code works unchanged with governance features added

**🎯 Next Step:** Ready for advanced features? Continue to Level 2!

---

### 🟡 Level 2: Advanced Observability (1 hour total)
**🎯 Goal:** Build production-ready workflows with commercial platform features  
**🏷️ Best For:** Teams ready to optimize costs and implement advanced governance

**🎓 What You'll Learn:**
- How to integrate Traceloop commercial platform with governance tracking
- Advanced multi-provider observability with unified cost intelligence
- Cost optimization strategies based on detailed usage analytics  
- Enterprise-grade governance patterns for compliance and audit

**📝 Examples to Run:**

4. **[traceloop_platform.py](traceloop_platform.py)** (30 minutes)
   - **Commercial platform integration** with governance enhancement
   - Advanced insights and analytics with team collaboration features
   - Enterprise observability with automated governance policies
   - See the value of upgrading from open-source to commercial features

5. **[advanced_observability.py](advanced_observability.py)** (30 minutes)
   - **Multi-provider governance** with unified cost tracking
   - Complex workflow tracing with detailed cost analysis and optimization
   - Advanced patterns for A/B testing with governance attribution
   - Cost-performance optimization recommendations

**✅ Level 2 Success Criteria:**
- [ ] You can track costs across multiple AI providers with unified governance
- [ ] Your team can make optimization decisions based on cost/performance data  
- [ ] You have advanced observability workflows with governance automation
- [ ] You understand when to upgrade to Traceloop commercial platform

**🎯 Next Step:** Ready for enterprise deployment? Continue to Level 3!

---

### 🔴 Level 3: Enterprise Governance (4+ hours total)
**🎯 Goal:** Master production-grade deployment with enterprise governance  
**🏷️ Best For:** Production systems requiring compliance, high-availability, and enterprise scale

**🎓 What You'll Learn:**
- Production deployment patterns with high-availability and disaster recovery
- Enterprise compliance monitoring with automated audit trails
- Advanced error handling and recovery strategies for production systems
- Multi-region governance with unified observability and cost intelligence

**📝 Examples to Run:**

6. **[production_patterns.py](production_patterns.py)** (3+ hours)
   - **Enterprise deployment patterns** with high-availability architecture
   - Multi-region governance with automatic failover and disaster recovery
   - Production monitoring with cost intelligence, alerts, and compliance automation
   - Advanced governance policies for SOC2, GDPR, and HIPAA compliance

7. **[error_scenarios_demo.py](error_scenarios_demo.py)** (30 minutes)
   - **Comprehensive error handling** and recovery demonstration
   - Production-grade failure scenarios with automatic remediation
   - Robust governance even during system failures and degraded performance
   - Actionable diagnostics and troubleshooting for production issues

**✅ Level 3 Success Criteria:**
- [ ] You can deploy multi-region governance systems with automatic failover
- [ ] Your organization has automated compliance monitoring and audit trails
- [ ] You have production-grade cost intelligence dashboards and alerting
- [ ] You understand enterprise governance patterns and can train your team

**🏆 Congratulations!** You've mastered enterprise-grade LLM governance with observability!

---

## 🏃 Running Examples

### Option 1: Individual Examples (Recommended for Learning)

```bash
# 🎯 Level 1: Getting Started (15 minutes total)
python setup_validation.py      # ⭐ Always start here
python basic_tracking.py        # See governance in action  
python auto_instrumentation.py  # Zero-code integration

# 📊 Level 2: Advanced Observability (1 hour total)
python traceloop_platform.py    # Commercial platform features
python advanced_observability.py # Advanced patterns

# 🏭 Level 3: Enterprise Governance (4+ hours total)
python production_patterns.py   # Enterprise deployment
```

### Option 2: Complete Suite (For Comprehensive Evaluation)

```bash
# Run all examples with validation (~20 minutes active time)
./run_all_examples.sh
```

This script includes progress tracking, error handling, and comprehensive reporting.

---

## 🎯 Industry-Specific Use Cases

### 🏦 Financial Services
- **Compliance:** SOC2, PCI DSS audit trails for all LLM operations
- **Cost Control:** Department-level budget attribution and enforcement
- **Risk Management:** Policy compliance for customer data processing
- **Examples:** Start with `traceloop_platform.py` for compliance tracking

### 🏥 Healthcare
- **HIPAA Compliance:** Encrypted governance attributes and audit logs
- **Cost Attribution:** Patient care vs. research cost separation
- **Quality Assurance:** Evaluation workflows with governance oversight
- **Examples:** Focus on `production_patterns.py` for compliance automation

### 🏢 Enterprise SaaS
- **Customer Attribution:** Per-customer cost tracking and billing
- **Team Governance:** Department-level budget controls and reporting
- **Feature Development:** A/B testing with cost attribution
- **Examples:** `advanced_observability.py` for cost-optimized customer experiences

### 🎓 Research & Education
- **Grant Tracking:** Research project cost attribution and reporting
- **Collaboration:** Multi-team governance with shared resources
- **Evaluation:** Research quality metrics with cost tracking
- **Examples:** `basic_tracking.py` for simple project attribution

---

## 🔧 Advanced Troubleshooting

### Setup Issues
**❌ "Command not found: python"**
```bash
# On macOS/Linux, try python3
python3 setup_validation.py
```

**❌ "OpenLLMetry not found"**
```bash
# Install OpenLLMetry directly
pip install openllmetry
# Or reinstall with all dependencies
pip install genops[traceloop]
```

**❌ "No LLM provider API keys found"**
```bash
# Verify at least one provider is configured
echo $OPENAI_API_KEY       # Should be set if using OpenAI
echo $ANTHROPIC_API_KEY    # Should be set if using Anthropic
```

### Advanced Troubleshooting
**❌ Governance integration issues:**
```bash
# Enable detailed logging for diagnosis
export GENOPS_LOG_LEVEL=DEBUG
python basic_tracking.py
```

**❌ OpenLLMetry connectivity problems:**
```bash
# Test OpenLLMetry instrumentation
python -c "import openllmetry; openllmetry.instrument(); print('✅ Ready')"
```

---

## 🆘 Need Help?

### 📚 Documentation
- **[5-Minute Quickstart Guide](../../docs/traceloop-quickstart.md)** - Fastest way to get started
- **[Complete Integration Guide](../../docs/integrations/traceloop.md)** - Comprehensive reference
- **[CLAUDE.md](../../CLAUDE.md)** - Development standards and patterns

### 💬 Community Support  
- **[GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)** - Bug reports and feature requests
- **[GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)** - Community questions and sharing

### 🚀 Professional Services
For enterprise deployments, custom integrations, or professional services, contact our team for dedicated support.

---

## 🌟 What's Next?

### After Level 1 (Basic Understanding):
1. **Integrate with your application:** Use patterns from `basic_tracking.py`
2. **Set up team attribution:** Configure governance attributes for your teams
3. **Monitor cost trends:** Watch your observability dashboard for governance insights

### After Level 2 (Advanced Features):
1. **Evaluate Traceloop platform:** Consider commercial platform for advanced insights
2. **Optimize operations:** Use cost intelligence from `advanced_observability.py`
3. **Set up advanced monitoring:** Create governance-aware observability workflows

### After Level 3 (Enterprise Ready):
1. **Production deployment:** Follow `production_patterns.py` guidance
2. **Enterprise integration:** Connect to your existing observability stack
3. **Team training:** Share governance patterns across your organization

---

**🎉 Ready to enhance your OpenLLMetry observability with GenOps governance?**

**Start your journey:** `python setup_validation.py`