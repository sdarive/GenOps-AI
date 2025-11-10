# CrewAI + GenOps: 5-Minute Quickstart

Add complete cost tracking and governance to your CrewAI multi-agent systems in under 5 minutes.

## 🤔 What is GenOps?

GenOps adds the missing tracking layer to your AI stack. Think of it as **OpenTelemetry for AI** - it automatically tracks costs, performance, and usage across all your AI tools without changing your existing code.

**For CrewAI specifically, GenOps gives you:**
- 💰 **Cost tracking** across OpenAI, Anthropic, Google, etc. 
- ⚡ **Performance monitoring** for agents and workflows
- 👥 **Team attribution** so you know who's using what
- 🚨 **Budget controls** to prevent surprise bills
- 📊 **Multi-agent insights** like bottlenecks and collaboration patterns

**The best part:** Your existing CrewAI code doesn't change. Just add 2 lines and get automatic tracking.

## ⚡ Zero-Code Setup (2 minutes)

### Prerequisites
- Python 3.9+ 
- An OpenAI API key (or other AI provider)

### 1. Install
```bash
pip install genops-ai[crewai] crewai
```

### 2. Set API Key  
```bash
export OPENAI_API_KEY="your_key_here"
```

### 3. Add 2 Lines to Your Code
```python
from genops.providers.crewai import auto_instrument

# Add these 2 lines to your existing CrewAI code
auto_instrument(
    team="your-team", 
    project="your-project"
)

# Your existing CrewAI code works unchanged!
# crew.kickoff() now has automatic tracking
```

**That's it!** Every crew execution now includes cost tracking, performance monitoring, and team attribution.

---

## 🚀 Complete Working Example (3 minutes)

Copy-paste this complete example to see it working:

```python
#!/usr/bin/env python3
"""5-Minute CrewAI + GenOps Demo"""

from crewai import Agent, Task, Crew
from genops.providers.crewai import auto_instrument

# 1. Enable GenOps (adds automatic tracking)
auto_instrument(
    team="demo-team",
    project="quickstart",
    daily_budget_limit=5.0  # $5 daily limit
)

# 2. Your existing CrewAI code (unchanged!)
researcher = Agent(
    role='Research Analyst',
    goal='Research AI trends',
    backstory='Expert in AI research'
)

writer = Agent(
    role='Writer', 
    goal='Write clear summaries',
    backstory='Technical writer'
)

# 3. Create tasks
research_task = Task(
    description='Research latest AI developments in 2024',
    agent=researcher
)

write_task = Task(
    description='Write a brief summary of the research',
    agent=writer
)

# 4. Create and run crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task]
)

print("🚀 Starting crew...")
result = crew.kickoff()

print("\n✅ Done!")
print(f"📝 Result: {result[:100]}...")  # Show first 100 chars

# 5. Check cost (automatically tracked)
from genops.providers.crewai import get_cost_summary
costs = get_cost_summary()
print(f"💰 Cost: ${costs.get('total_cost', 0):.4f}")
```

**Run it:**
```bash
python demo.py
```

**Expected output:**
```
🚀 Starting crew...
> Entering new CrewAI crew: Research Analyst
> Finished chain.
> Entering new CrewAI crew: Writer  
> Finished chain.
✅ Done!
📝 Result: Based on my research, AI developments in 2024 include...
💰 Cost: $0.0245
```

---

## ✅ Validation (30 seconds)

Verify your setup works:

```bash
python -c "
from genops.providers.crewai import validate_crewai_setup, print_validation_result
result = validate_crewai_setup(quick=True)
print_validation_result(result)
"
```

**Success output:**
```
✅ CrewAI installation: Found crewai 0.x.x
✅ GenOps integration: Available 
✅ API keys: OpenAI configured
✅ Environment: Ready for multi-agent tracking
```

---

## 🎯 What You Just Got

With those 2 lines of code (`auto_instrument()`), every crew execution now includes:

- **💰 Cost tracking** across OpenAI, Anthropic, Google, etc.
- **⚡ Performance monitoring** for agents and tasks
- **🎯 Governance telemetry** with team/project attribution  
- **📊 Multi-agent insights** like collaboration patterns
- **🚨 Budget controls** and spending limits
- **📈 Usage analytics** exportable to any observability platform

---

## 🚀 Next Steps (Optional)

**Explore More (5 minutes each):**
- **Manual Control**: [examples/crewai/basic_crew_tracking.py](../../examples/crewai/basic_crew_tracking.py)
- **Cost Analysis**: [examples/crewai/multi_agent_cost_aggregation.py](../../examples/crewai/multi_agent_cost_aggregation.py)  
- **Performance**: [examples/crewai/performance_optimization.py](../../examples/crewai/performance_optimization.py)
- **Production**: [examples/crewai/production_deployment_patterns.py](../../examples/crewai/production_deployment_patterns.py)

**Integration Guide**: [examples/crewai/README.md](../../examples/crewai/README.md)

---

## 🔧 Need Help?

**Common Issues:**

1. **No API key**: `export OPENAI_API_KEY="your_key"`
2. **CrewAI not found**: `pip install crewai`  
3. **GenOps not found**: `pip install genops-ai[crewai]`

**Detailed diagnostics:**
```bash
cd examples/crewai && python setup_validation.py
```

---

**Ready in 5 minutes or less!** 🎉

*Time-to-value validated with new developers. Questions? See [troubleshooting](../../examples/crewai/README.md#troubleshooting).*