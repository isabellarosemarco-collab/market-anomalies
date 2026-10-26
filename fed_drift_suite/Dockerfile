# Image for LAYER 2 ONLY — the commercial API.
#
# The personal trader is deliberately not the container's job. It holds brokerage credentials that can
# move real money, while this image is the thing exposed to the internet and handed to clients; those
# two should not share a blast radius, a restart policy, or a log stream. Run the trader on your own
# machine (`python personal/runner.py`), deploy this.
#
# The container runs as a non-root user and the image contains no secrets. API keys, the Alpaca keys
# (unused here) and webhook secrets all arrive as environment variables from the platform's secret
# store at run time, which keeps them out of the image layers, out of git, and out of `docker history`.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Dependencies first, so a code change does not re-resolve the whole tree on every build.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py ./
COPY core/ ./core/
COPY commercial/ ./commercial/
COPY personal/ ./personal/
COPY test_suite.py README.md ./

# Writable location for the subscription store. Mount a volume here in production: the file holds
# webhook signing secrets, and a redeploy otherwise loses every client's registration.
RUN mkdir -p /data && chmod 700 /data
ENV SUBS_PATH=/data/subscriptions.json

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app /data
USER appuser

EXPOSE 8000

# `/health` is unauthenticated precisely so this works without baking a key into the image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,os,sys; \
sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",8000)}/health', timeout=4).status==200 else 1)"

# One worker on purpose. The webhook watcher fires at 15:00 ET from an in-process loop, so N workers
# would mean N deliveries of the same signal. Scale out only after moving the watcher to its own
# process or adding a shared lock.
CMD ["sh", "-c", "uvicorn commercial.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
