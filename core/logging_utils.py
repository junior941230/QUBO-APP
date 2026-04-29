from datetime import datetime

def log_step(message):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)

def parse_float_list(text, default):
    if not text or not text.strip():
        return list(default)
    try:
        return [float(x.strip()) for x in text.split(",") if x.strip()]
    except ValueError:
        log_step(f"[Parse] invalid float list '{text}', fallback to default {default}")
        return list(default)