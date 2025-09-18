# test_add.py — quick non-Streamlit test to exercise db functions
import traceback
import os, sys
from datetime import date

# adjust path if needed
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, ROOT)

print("Using python:", sys.executable)
print("Importing db module from:", ROOT)

try:
    from db import init_db, add_meal, get_logs_for_date, get_logs_between
    print("Imported db module OK")
except Exception as e:
    print("Failed to import db module:", e)
    traceback.print_exc()
    raise SystemExit(1)

try:
    # initialize (idempotent)
    init_db()
    print("init_db() ran")
except Exception as e:
    print("init_db() raised:", e)
    traceback.print_exc()

# Print attributes of db module (helps locate DB file)
print("\n--- db module attributes (dir) ---")
import inspect, db as dbmod
for name in ("__file__",):
    print(name, "=", getattr(dbmod, name, "N/A"))

# If db module exposes DB path variable, show it:
cands = ["DB_PATH", "DB_FILE", "DBNAME", "DATABASE", "DB"]
for cand in cands:
    if hasattr(dbmod, cand):
        print(f"{cand} =", getattr(dbmod, cand))

# Try adding a test meal
try:
    test_name = "TEST_ADD_APP"
    print("\nAdding test meal:", test_name)
    ok = add_meal(food_name=test_name, calories=123.0, protein=4.0, carbs=12.0, fat=3.0, serving="100 g", source="test_script")
    print("add_meal returned:", ok)
except Exception as e:
    print("add_meal threw exception:")
    traceback.print_exc()

# Query today's logs and print rows
try:
    logs = get_logs_for_date(date.today())
    print(f"\nget_logs_for_date({date.today()}) returned {len(logs) if logs else 0} rows")
    for r in logs[-10:]:
        print(r)
except Exception as e:
    print("get_logs_for_date raised:")
    traceback.print_exc()
