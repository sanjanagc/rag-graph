import os
import sys
from pathlib import Path

# Add project3_graph_rag to sys.path
proj_dir = Path(__file__).resolve().parent.parent
if str(proj_dir) not in sys.path:
    sys.path.insert(0, str(proj_dir))
