#!/usr/bin/env python3
"""
E8 Leech Lattice Framework - CLI Module
Entry point for setuptools console scripts
"""

import sys
import os

# Add the parent directory to the path to find the main CLI
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, parent_dir)

def main():
    """Main entry point for the CLI"""
    try:
        from e8leech_cli import main as cli_main
        cli_main()
    except ImportError:
        # Fallback to direct import
        import importlib.util
        cli_path = os.path.join(parent_dir, 'e8leech_cli.py')
        spec = importlib.util.spec_from_file_location("e8leech_cli", cli_path)
        cli_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli_module)
        cli_module.main()

if __name__ == '__main__':
    main()

