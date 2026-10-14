# CPU-only image for the Streamlit app. Works on any Docker host, including Hugging Face Spaces
# (SDK: Docker; app_port 7860). Build:  docker build -t voc-intel .   Run:  docker run -p 7860:7860 voc-intel
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    HF_HOME=/home/user/.cache/huggingface \
    VOC_DB_PATH=/home/user/app/data/voc.db

# libgomp1: OpenMP runtime used by torch / numba / hdbscan. build-essential: for any package
# without a prebuilt wheel for this platform (hdbscan is the usual one).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (Hugging Face Spaces runs containers as uid 1000).
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --user -r requirements.txt
ENV PATH="/home/user/.local/bin:${PATH}"

# Bake the models into the image so the first analysis or topic run doesn't download them.
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest')" \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY --chown=user . .

EXPOSE 7860
# The demo database in data/demo/ is copied to $VOC_DB_PATH on first start (see db.ensure_demo_db).
CMD ["sh", "-c", "streamlit run dashboard/app.py --server.port=${PORT} --server.address=0.0.0.0 --server.fileWatcherType=none"]
