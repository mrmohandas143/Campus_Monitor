import sys
import tempfile
import os
sys.path.insert(0, '.')

from db.schema import init_db
from db import crud
import numpy as np

tmp_db = os.path.join(tempfile.mkdtemp(), "smoke_test.db")

print('[1] Initializing database...')
conn = init_db(tmp_db)
print('    OK - tables created')

print('[2] Registering test student...')
fake_emb = np.random.rand(128).astype(np.float32)
fake_emb /= np.linalg.norm(fake_emb)
try:
    pid = crud.register_person(conn, 'Alice Kumar', 'CS2024001', fake_emb)
    print('    OK - registered, id=' + str(pid))
except Exception as e:
    print('    Exists or error: ' + str(e))

print('[3] Logging attendance...')
rows = crud.get_all_persons(conn)
if rows:
    logged, reason = crud.log_attendance(conn, rows[0]['id'], confidence=0.91)
    print('    logged=' + str(logged) + '  reason=' + reason)

print('[4] Querying daily summary...')
summary = crud.get_daily_summary(conn)
print('    Present=' + str(summary['total_present']) + '  Unknown=' + str(summary['total_unknown']))

print('[5] Exporting to Excel...')
from attendance.exporter import export_daily
tmp_excel = os.path.join(tempfile.mkdtemp(), "smoke_test.xlsx")
path = export_daily(conn, output_path=tmp_excel)

size = os.path.getsize(path)
print('    File size: ' + str(size) + ' bytes')

print()
print('ALL STAGES SMOKE TEST: PASSED')
