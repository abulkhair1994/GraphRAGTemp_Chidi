#!/usr/bin/env python3
"""
Convenience script to set up comprehensive vector indexes.

This script runs the comprehensive vector index creation from the scripts directory.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run the comprehensive vector index setup."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Path to the actual script
    script_path = project_root / "src" / "embeddings" / "create_comprehensive_vector_index.py"
    
    print("🚀 Setting up comprehensive educational vector indexes...")
    print(f"📁 Project root: {project_root}")
    print(f"📄 Running script: {script_path}")
    print("=" * 60)
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, str(script_path)], 
                              cwd=str(project_root),
                              check=True)
        
        print("\n" + "=" * 60)
        print("✅ Vector index setup completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running vector index setup: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 