"""
Root entry point forwarding to screen_analyzer.main.
Allows running: python main.py [args...]
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from screen_analyzer.main import main

if __name__ == "__main__":
    main()
