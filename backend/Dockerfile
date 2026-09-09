FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN --mount=type=cache,target=/var/lib/apt/lists \
    sed -i 's|http://deb.debian.org|https://mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Retries=3 -o Acquire::https::Timeout=30 update \
    && apt-get -o Acquire::Retries=3 -o Acquire::https::Timeout=30 install -y --no-install-recommends --only-upgrade \
      bsdutils libblkid1 liblastlog2-2 libmount1 libsmartcols1 libuuid1 login mount util-linux

RUN groupadd --system newrag && useradd --system --gid newrag --home /app newrag

COPY requirements-runtime.txt ./requirements.txt
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
# Runtime does not install packages. Removing pip also removes its vendored SBOM,
# which otherwise reports stale msgpack/setuptools components not imported by the app.
RUN python -m pip uninstall -y pip

COPY app ./app
# Offline research/evaluation scripts stay in the checkout, not the API image.
# Operational API and Celery entrypoints import only the application package.
RUN mkdir -p /app/data/uploads /app/data/pipeline && chown -R newrag:newrag /app

USER newrag
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/livez', timeout=2)"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
