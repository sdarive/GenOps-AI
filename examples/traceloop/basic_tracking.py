#!/usr/bin/env python3
"""
Basic OpenLLMetry + GenOps Tracking Example

This example demonstrates how to enhance OpenLLMetry observability with GenOps governance,
providing cost attribution, team tracking, and policy enforcement for your LLM operations.

About OpenLLMetry:
OpenLLMetry is an open-source observability framework that extends OpenTelemetry with 
LLM-specific instrumentation. GenOps enhances this foundation with governance intelligence.

Usage:
    python basic_tracking.py

Prerequisites:
    pip install genops[traceloop]  # Includes OpenLLMetry
    export OPENAI_API_KEY="your-openai-api-key"
    
    # Optional: For Traceloop commercial platform
    export TRACELOOP_API_KEY="your-traceloop-api-key"
"""

import os
import asyncio
from datetime import datetime
from typing import List, Dict, Any


def basic_openllmetry_with_genops():
    """
    Demonstrates basic OpenLLMetry instrumentation enhanced with GenOps governance.
    
    This example shows how GenOps adds cost attribution, team tracking, and 
    governance context to standard OpenLLMetry traces.
    """
    print("🔍 Basic OpenLLMetry + GenOps Tracking Example")
    print("=" * 50)
    
    try:
        # Import GenOps Traceloop adapter (built on OpenLLMetry)
        from genops.providers.traceloop import instrument_traceloop
        print("✅ GenOps Traceloop adapter loaded successfully")
        
        # Initialize with governance context
        adapter = instrument_traceloop(
            team="engineering",
            project="llm-chatbot",
            customer_id="demo-customer",
            environment="development",
            cost_center="rd-department"
        )
        print("✅ GenOps governance context configured")
        
    except ImportError as e:
        print(f"❌ Failed to import GenOps Traceloop adapter: {e}")
        print("💡 Fix: Run 'pip install genops[traceloop]'")
        return False
    
    try:
        # Import OpenAI for LLM calls
        import openai
        client = openai.OpenAI()
        print("✅ OpenAI client initialized")
        
    except ImportError:
        print("❌ OpenAI library not found")
        print("💡 Fix: Run 'pip install openai'")
        return False
    
    print("\n🚀 Running Enhanced LLM Operations...")
    print("-" * 40)
    
    # Example 1: Simple chat completion with governance
    print("\n1️⃣ Simple Chat Completion with Cost Attribution")
    try:
        with adapter.track_operation(
            operation_type="chat_completion",
            operation_name="basic_chat",
            tags={"use_case": "customer_support", "priority": "high"}
        ) as span:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What are the benefits of LLM observability?"}
                ],
                max_tokens=150
            )
            
            # GenOps automatically captures cost and governance data
            content = response.choices[0].message.content
            print(f"✅ Response generated with governance tracking")
            print(f"📝 Content: {content[:100]}...")
            
            # Access governance-enhanced metrics
            metrics = span.get_metrics()
            print(f"💰 Estimated cost: ${metrics.get('estimated_cost', 'N/A')}")
            print(f"🏷️ Team attribution: {metrics.get('team', 'N/A')}")
            print(f"📊 Tokens used: {metrics.get('total_tokens', 'N/A')}")
            
    except Exception as e:
        print(f"❌ Chat completion failed: {e}")
        print("🔧 Troubleshooting:")
        print("   • Check API key: echo $OPENAI_API_KEY")
        print("   • Verify network connectivity")
        print("   • Check API rate limits and quotas")
        if "api key" in str(e).lower():
            print("   💡 API Key Issue: Set OPENAI_API_KEY environment variable")
        elif "rate limit" in str(e).lower():
            print("   💡 Rate Limit: Wait before retrying or upgrade API plan")
        elif "network" in str(e).lower() or "connection" in str(e).lower():
            print("   💡 Network Issue: Check internet connection and firewall settings")
        return False
    
    # Example 2: Batch operations with team attribution
    print("\n2️⃣ Batch Operations with Team Cost Tracking")
    try:
        batch_requests = [
            "Explain machine learning in one sentence.",
            "What is the capital of France?",
            "How do neural networks work?"
        ]
        
        with adapter.track_operation(
            operation_type="batch_processing",
            operation_name="batch_qa",
            tags={"batch_size": len(batch_requests), "team": "engineering"}
        ) as batch_span:
            
            batch_costs = []
            for i, request in enumerate(batch_requests):
                with adapter.track_operation(
                    operation_type="individual_completion",
                    operation_name=f"batch_item_{i+1}",
                    parent_span=batch_span
                ) as item_span:
                    
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": request}],
                        max_tokens=50
                    )
                    
                    metrics = item_span.get_metrics()
                    cost = metrics.get('estimated_cost', 0.0)
                    batch_costs.append(cost)
                    
                    print(f"   ✅ Request {i+1}: ${cost:.6f}")
            
            total_cost = sum(batch_costs)
            print(f"💰 Total batch cost: ${total_cost:.6f}")
            print(f"🏷️ Cost attributed to team: engineering")
            
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        print("🔧 Batch Processing Troubleshooting:")
        print("   • Check if individual requests exceed rate limits")
        print("   • Verify batch size is reasonable (<100 requests)")
        print("   • Consider adding delays between requests")
        if "rate limit" in str(e).lower():
            print("   💡 Rate Limit: Implement exponential backoff or reduce batch size")
        elif "timeout" in str(e).lower():
            print("   💡 Timeout: Increase timeout or process in smaller batches")
        return False
    
    # Example 3: Function calling with governance
    print("\n3️⃣ Function Calling with Governance Tracking")
    try:
        # Define a function for the LLM to call
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate_cost_savings",
                    "description": "Calculate potential cost savings from LLM optimization",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "current_monthly_cost": {
                                "type": "number",
                                "description": "Current monthly LLM costs in USD"
                            },
                            "optimization_percentage": {
                                "type": "number",
                                "description": "Expected percentage of cost reduction (0-100)"
                            }
                        },
                        "required": ["current_monthly_cost", "optimization_percentage"]
                    }
                }
            }
        ]
        
        def calculate_cost_savings(current_monthly_cost: float, optimization_percentage: float) -> dict:
            """Calculate cost savings from optimization."""
            savings = current_monthly_cost * (optimization_percentage / 100)
            annual_savings = savings * 12
            return {
                "monthly_savings": savings,
                "annual_savings": annual_savings,
                "optimization_percentage": optimization_percentage
            }
        
        with adapter.track_operation(
            operation_type="function_calling",
            operation_name="cost_optimization_analysis",
            tags={"function_type": "cost_analysis", "team": "finops"}
        ) as func_span:
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "user", 
                    "content": "I'm spending $1000 per month on LLM operations. Calculate potential savings with 30% optimization."
                }],
                tools=tools,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                function_args = eval(tool_call.function.arguments)
                
                # Execute the function
                result = calculate_cost_savings(**function_args)
                
                print(f"✅ Function called: {tool_call.function.name}")
                print(f"💰 Monthly savings: ${result['monthly_savings']:.2f}")
                print(f"📈 Annual savings: ${result['annual_savings']:.2f}")
                print(f"🏷️ Analysis attributed to team: finops")
                
                # Add function result to governance tracking
                func_span.add_attributes({
                    "function.name": tool_call.function.name,
                    "function.monthly_savings": result['monthly_savings'],
                    "function.annual_savings": result['annual_savings']
                })
            
    except Exception as e:
        print(f"❌ Function calling failed: {e}")
        return False
    
    return True


def demonstrate_governance_features():
    """Demonstrate specific GenOps governance features."""
    print("\n🛡️ GenOps Governance Features Demo")
    print("-" * 35)
    
    try:
        from genops.providers.traceloop import instrument_traceloop
        
        # Initialize with strict governance policies
        adapter = instrument_traceloop(
            team="compliance-team",
            project="sensitive-data-processing",
            environment="production",
            enable_cost_alerts=True,
            max_operation_cost=0.10,  # $0.10 limit per operation
            require_approval_above=0.05  # Require approval above $0.05
        )
        
        print("✅ Governance policies configured:")
        print("   • Cost alerts: Enabled")
        print("   • Max operation cost: $0.10")
        print("   • Approval required above: $0.05")
        
        # Test governance enforcement
        import openai
        client = openai.OpenAI()
        
        with adapter.track_operation(
            operation_type="governance_test",
            operation_name="policy_enforcement_demo"
        ) as span:
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "user", 
                    "content": "Write a short summary of LLM governance best practices."
                }],
                max_tokens=100
            )
            
            metrics = span.get_metrics()
            cost = metrics.get('estimated_cost', 0.0)
            
            if cost > 0.05:
                print(f"⚠️ Cost threshold exceeded: ${cost:.6f}")
                print("🛡️ Governance policy would require approval in production")
            else:
                print(f"✅ Operation within cost limits: ${cost:.6f}")
            
            print(f"📊 Governance context captured:")
            print(f"   • Team: {metrics.get('team')}")
            print(f"   • Project: {metrics.get('project')}")
            print(f"   • Environment: {metrics.get('environment')}")
            
    except Exception as e:
        print(f"❌ Governance demo failed: {e}")
        return False
    
    return True


def show_openllmetry_integration():
    """Show how GenOps integrates with OpenLLMetry standards."""
    print("\n🔗 OpenLLMetry Integration Details")
    print("-" * 35)
    
    try:
        # Import OpenLLMetry directly to show integration
        import openllmetry
        from opentelemetry import trace
        
        print("✅ OpenLLMetry foundation:")
        print(f"   • OpenLLMetry version: {getattr(openllmetry, '__version__', 'unknown')}")
        print("   • Built on OpenTelemetry standards")
        print("   • Vendor-neutral observability")
        
        # Show how GenOps enhances the OpenLLMetry tracer
        from genops.providers.traceloop import get_enhanced_tracer
        
        tracer = get_enhanced_tracer()
        print("✅ GenOps enhancements:")
        print("   • Automatic cost calculation")
        print("   • Team and project attribution")
        print("   • Policy enforcement")
        print("   • Budget tracking")
        
        # Create an enhanced span
        with tracer.start_span("genops_enhanced_operation") as span:
            span.set_attribute("genops.team", "engineering")
            span.set_attribute("genops.project", "demo")
            span.set_attribute("genops.cost.currency", "USD")
            span.set_attribute("genops.cost.amount", 0.002)
            
            print("✅ Enhanced span created with GenOps attributes")
            print("   • Standard OpenTelemetry span")
            print("   • Enhanced with governance attributes")
            print("   • Compatible with all OpenTelemetry backends")
    
    except Exception as e:
        print(f"❌ Integration demo failed: {e}")
        return False
    
    return True


async def main():
    """Main execution function."""
    print("🚀 Starting OpenLLMetry + GenOps Basic Tracking Demo")
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check prerequisites
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not found")
        print("💡 Set your OpenAI API key: export OPENAI_API_KEY='your-key'")
        return False
    
    # Run examples
    success = True
    
    # Basic tracking examples
    if not basic_openllmetry_with_genops():
        success = False
    
    # Governance features
    if success and not demonstrate_governance_features():
        success = False
    
    # OpenLLMetry integration details
    if success and not show_openllmetry_integration():
        success = False
    
    if success:
        print("\n" + "🌟" * 50)
        print("🎉 OpenLLMetry + GenOps Basic Tracking Demo Complete!")
        print("\n📊 What You've Accomplished:")
        print("   ✅ Enhanced OpenLLMetry with governance intelligence")
        print("   ✅ Automatic cost attribution and team tracking")
        print("   ✅ Policy enforcement and budget monitoring")
        print("   ✅ Compatible with all OpenTelemetry backends")
        
        print("\n🔍 Your Enhanced Observability Stack:")
        print("   • OpenLLMetry: Open-source LLM observability foundation")
        print("   • GenOps: Governance, cost intelligence, and policy enforcement")
        print("   • OpenTelemetry: Industry-standard observability protocol")
        print("   • Vendor-neutral: Works with Datadog, Honeycomb, Grafana, etc.")
        
        print("\n📚 Next Steps:")
        print("   • Run 'python auto_instrumentation.py' for zero-code integration")
        print("   • Run 'python traceloop_platform.py' for commercial platform features")
        print("   • Explore advanced patterns with 'python advanced_observability.py'")
        
        print("\n💡 Quick Integration:")
        print("   Add this to your existing OpenLLMetry code:")
        print("   ```python")
        print("   from genops.providers.traceloop import instrument_traceloop")
        print("   adapter = instrument_traceloop(team='your-team', project='your-project')")
        print("   # Your existing OpenLLMetry code works unchanged!")
        print("   ```")
        
        print("🌟" * 50)
    else:
        print("\n❌ Demo encountered errors. Please check the output above.")
    
    return success


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())