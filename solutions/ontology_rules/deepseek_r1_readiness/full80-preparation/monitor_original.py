"""Single local read-only observer; monitor failures never interrupt remote inference."""
from pathlib import Path
import fcntl,time
from poll_original import main
p=Path(__file__).parent;handle=(p/'monitor.lock').open('a+');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
start=time.monotonic()
while time.monotonic()-start<10980:
 if main():break
 time.sleep(60)
