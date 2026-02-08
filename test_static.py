#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/brian.day/git/crsd-inspector/.venv/lib/python3.11/site-packages')

try:
    import staticdash
    print("staticdash imported successfully")
    print(f"Location: {staticdash.__file__}")
    print(f"Available: {[x for x in dir(staticdash) if not x.startswith('_')]}")
    
    # Try creating a report
    report = staticdash.Report("Test")
    print(f"Report methods: {[x for x in dir(report) if not x.startswith('_')]}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
