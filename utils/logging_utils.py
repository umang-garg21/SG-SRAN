import sys
import datetime


def log(msg: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sys.stdout.write(f"[{timestamp}] {msg}\n")
    sys.stdout.flush()
