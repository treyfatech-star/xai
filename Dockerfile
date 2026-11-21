# procure-xai: one image for data build, training, API, dashboard and experiments.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

# No apt packages needed: the XGBoost and scikit-learn wheels bundle their own OpenMP runtime.
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md DECISIONS.md Makefile ./
COPY config ./config
COPY src ./src
COPY dashboard ./dashboard
COPY experiments ./experiments
COPY tests ./tests
COPY docs ./docs
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN pip install --no-deps -e . && chmod +x /usr/local/bin/entrypoint.sh \
    && mkdir -p data/raw data/interim data/processed models experiments/results

EXPOSE 5000 8501
ENTRYPOINT ["entrypoint.sh"]
CMD ["dashboard"]
