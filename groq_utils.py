# I designed this file to handle groq ratelimiting error with exponential backoff.
import time


def call_with_retry(fn, *args, max_retries: int = 6, base_delay: float = 5.0, **kwargs):
    last_err = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            status = getattr(e, "status_code", None)
            is_rate_limit = status == 429 or "429" in str(e) or "rate_limit" in str(e).lower()
            if not is_rate_limit:
                raise
            retry_after = getattr(getattr(e, "response", None), "headers", {}).get("retry-after")
            delay = float(retry_after) if retry_after else base_delay * (2 ** attempt)
            print(f"  [rate limited, attempt {attempt+1}/{max_retries}] waiting {delay:.0f}s ...")
            time.sleep(delay)
    raise last_err