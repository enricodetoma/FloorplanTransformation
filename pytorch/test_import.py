import sys
import os

# Add the current directory to the Python path
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)

try:
    from models.model import Model
    print("Import successful")
except ImportError as e:
    print("Import failed: {}".format(e))
    print("Current PYTHONPATH:", sys.path)
