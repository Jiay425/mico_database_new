"""Start the local Spring Boot UI only after the Meta2DB direct import succeeds."""
from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path

pid = int(sys.argv[1])
project = Path(__file__).resolve().parents[1]
log = project / "model" / "meta2db_species_direct_import.log"
app_log = project / "logs" / "meta2db-ui-startup.log"
app_error = project / "logs" / "meta2db-ui-startup.error.log"
while True:
    tasklist = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    if str(pid) not in tasklist:
        break
    time.sleep(10)
if "Completed direct Meta2DB import:" in log.read_text(encoding="utf-8", errors="replace"):
    with app_log.open("w", encoding="utf-8") as out, app_error.open("w", encoding="utf-8") as err:
        subprocess.Popen([str(project / "mvnw.cmd"), "spring-boot:run"], cwd=project, stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
