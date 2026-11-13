FROM python:3.12.7-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY run_agent.py ./
COPY harness ./harness
COPY solvers ./solvers
COPY tools/collect_harness_report.py ./tools/collect_harness_report.py
# nothing is downloaded at run time; container runs read-only, writes only to /tmp and the tasks dir
ENTRYPOINT ["python", "/app/run_agent.py"]
