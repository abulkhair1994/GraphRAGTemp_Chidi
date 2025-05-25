#!/usr/bin/env python3
"""
Auto-Repair Utility for GraphRAG Systems
========================================

This module provides automatic detection and repair of common GraphRAG issues,
particularly vector index embedding dimension mismatches.

Usage:
    from src.utils.auto_repair import AutoRepair
    
    # Create auto-repair instance
    repair = AutoRepair()
    
    # Check and fix issues automatically
    repair.check_and_fix_all()
    
    # Or check specific issues
    repair.check_vector_indexes()
"""

import os
import sys
import subprocess
from typing import Dict, List, Any, Optional
from pathlib import Path

class AutoRepair:
    """Automatic repair utility for GraphRAG systems"""
    
    def __init__(self, project_root: Optional[str] = None):
        """Initialize auto-repair utility
        
        Args:
            project_root: Path to project root (auto-detected if None)
        """
        if project_root is None:
            # Auto-detect project root
            current_file = Path(__file__).resolve()
            self.project_root = current_file.parent.parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.auto_fix_script = self.project_root / "auto_fix_vector_indexes.py"
        
    def check_vector_indexes(self) -> Dict[str, Any]:
        """Check and fix vector index issues automatically
        
        Returns:
            Dictionary with repair results
        """
        print("🔧 Auto-Repair: Checking vector indexes...")
        
        if not self.auto_fix_script.exists():
            return {
                "success": False,
                "error": f"Auto-fix script not found: {self.auto_fix_script}"
            }
        
        try:
            # Run the auto-fix script
            result = subprocess.run(
                [sys.executable, str(self.auto_fix_script)],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print("✅ Auto-Repair: Vector indexes fixed successfully!")
                return {
                    "success": True,
                    "message": "Vector indexes repaired",
                    "output": result.stdout
                }
            else:
                print(f"❌ Auto-Repair: Vector index fix failed (exit code: {result.returncode})")
                return {
                    "success": False,
                    "error": f"Fix script failed: {result.stderr}",
                    "output": result.stdout
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Auto-fix script timed out after 5 minutes"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to run auto-fix script: {e}"
            }
    
    def check_and_fix_all(self) -> Dict[str, Any]:
        """Check and fix all known issues automatically
        
        Returns:
            Dictionary with overall repair results
        """
        print("🚀 Auto-Repair: Starting comprehensive system check...")
        
        results = {
            "vector_indexes": self.check_vector_indexes(),
            "overall_success": True,
            "issues_fixed": 0,
            "issues_failed": 0
        }
        
        # Count successes and failures
        for check_name, check_result in results.items():
            if check_name.startswith("overall_") or check_name.endswith("_fixed") or check_name.endswith("_failed"):
                continue
                
            if isinstance(check_result, dict):
                if check_result.get("success", False):
                    results["issues_fixed"] += 1
                else:
                    results["issues_failed"] += 1
                    results["overall_success"] = False
        
        if results["overall_success"]:
            print(f"🎉 Auto-Repair: All systems operational! Fixed {results['issues_fixed']} issues.")
        else:
            print(f"⚠️ Auto-Repair: {results['issues_failed']} issues remain unfixed.")
        
        return results
    
    def is_repair_needed(self) -> bool:
        """Check if any repairs are needed without applying fixes
        
        Returns:
            True if repairs are needed, False otherwise
        """
        # This is a lightweight check - you could implement specific
        # detection logic here without running the full repair
        return True  # For now, always check
    
    def get_repair_status(self) -> Dict[str, Any]:
        """Get current repair status without making changes
        
        Returns:
            Dictionary with current system status
        """
        return {
            "auto_fix_script_exists": self.auto_fix_script.exists(),
            "project_root": str(self.project_root),
            "repair_available": True
        }


def auto_repair_on_failure(func):
    """Decorator to automatically repair system on GraphRAG failures
    
    Usage:
        @auto_repair_on_failure
        def my_graphrag_function():
            # Your GraphRAG code here
            pass
    """
    def wrapper(*args, **kwargs):
        try:
            # Try to run the function normally
            return func(*args, **kwargs)
        except Exception as e:
            # If it fails, try auto-repair
            print(f"🔧 Auto-Repair: Function {func.__name__} failed, attempting repair...")
            print(f"   Error: {e}")
            
            repair = AutoRepair()
            repair_result = repair.check_and_fix_all()
            
            if repair_result["overall_success"]:
                print("✅ Auto-Repair: System repaired, retrying function...")
                # Retry the function after repair
                return func(*args, **kwargs)
            else:
                print("❌ Auto-Repair: Could not repair system automatically")
                raise e
    
    return wrapper


# Convenience function for quick repairs
def quick_repair() -> bool:
    """Quick repair function for immediate use
    
    Returns:
        True if repair was successful, False otherwise
    """
    repair = AutoRepair()
    result = repair.check_and_fix_all()
    return result["overall_success"]


if __name__ == "__main__":
    # Allow running this module directly for testing
    repair = AutoRepair()
    result = repair.check_and_fix_all()
    sys.exit(0 if result["overall_success"] else 1) 