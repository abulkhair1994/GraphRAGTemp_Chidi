#!/usr/bin/env python3
"""
Auto-Repair Integration Demo
===========================

This example demonstrates how to integrate automatic repair functionality
into your GraphRAG applications to handle vector index issues automatically.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.auto_repair import AutoRepair, auto_repair_on_failure, quick_repair

def demo_basic_auto_repair():
    """Demonstrate basic auto-repair functionality"""
    print("🔧 Demo: Basic Auto-Repair")
    print("=" * 40)
    
    # Create auto-repair instance
    repair = AutoRepair()
    
    # Check current status
    status = repair.get_repair_status()
    print(f"📊 Repair Status: {status}")
    
    # Run comprehensive check and fix
    result = repair.check_and_fix_all()
    print(f"🎯 Repair Result: {result['overall_success']}")
    print(f"   Issues Fixed: {result['issues_fixed']}")
    print(f"   Issues Failed: {result['issues_failed']}")

@auto_repair_on_failure
def demo_graphrag_function_with_auto_repair():
    """Demonstrate a GraphRAG function with automatic repair on failure"""
    print("\n🤖 Demo: GraphRAG Function with Auto-Repair Decorator")
    print("=" * 50)
    
    # Simulate a GraphRAG operation that might fail due to vector index issues
    try:
        # This would be your actual GraphRAG code
        print("✅ GraphRAG operation completed successfully!")
        return "Success"
    except Exception as e:
        # The decorator will catch this and attempt auto-repair
        raise e

def demo_quick_repair():
    """Demonstrate quick repair function"""
    print("\n⚡ Demo: Quick Repair Function")
    print("=" * 35)
    
    # Use the convenience function for immediate repair
    success = quick_repair()
    if success:
        print("✅ Quick repair completed successfully!")
    else:
        print("❌ Quick repair failed")

def demo_manual_integration():
    """Demonstrate manual integration into existing code"""
    print("\n🔧 Demo: Manual Integration Example")
    print("=" * 40)
    
    def your_existing_graphrag_function():
        """Your existing GraphRAG function"""
        # Simulate potential vector index issues
        print("🔍 Running GraphRAG query...")
        
        # Check if repair is needed and apply if necessary
        repair = AutoRepair()
        if repair.is_repair_needed():
            print("⚠️  Potential issues detected, running auto-repair...")
            repair_result = repair.check_and_fix_all()
            
            if not repair_result["overall_success"]:
                raise Exception("Auto-repair failed, cannot proceed")
        
        print("✅ GraphRAG query completed successfully!")
        return "Query results here"
    
    # Run the function
    result = your_existing_graphrag_function()
    print(f"📊 Result: {result}")

def main():
    """Run all auto-repair demos"""
    print("🚀 GraphRAG Auto-Repair Integration Demos")
    print("=" * 50)
    
    # Demo 1: Basic auto-repair
    demo_basic_auto_repair()
    
    # Demo 2: Decorator-based auto-repair
    demo_graphrag_function_with_auto_repair()
    
    # Demo 3: Quick repair
    demo_quick_repair()
    
    # Demo 4: Manual integration
    demo_manual_integration()
    
    print("\n🎉 All demos completed!")
    print("\n📝 Integration Instructions:")
    print("   1. Import: from src.utils.auto_repair import AutoRepair, auto_repair_on_failure")
    print("   2. Use @auto_repair_on_failure decorator on GraphRAG functions")
    print("   3. Or call AutoRepair().check_and_fix_all() manually when needed")
    print("   4. The system will automatically detect and fix vector index issues")

if __name__ == "__main__":
    main() 