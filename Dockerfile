# Azure App Service for Containers / Azure Container Apps - portable to
# either without changes (both just run this image and route traffic to
# the exposed port). Pinned to the exact Python version this project is
# developed and tested against locally (3.13), rather than trusting
# whatever a platform's native "Python runtime" buildpack happens to ship,
# to avoid a version-mismatch surprise in production.
FROM python:3.13-slim

# poppler-utils: required for PDF uploads (services/preprocessing/
# pdf_processing.py calls pdftoppm via pdf2image) - a real, used feature,
# not optional. libglib2.0-0/libsm6/libxext6/libxrender1: opencv-python-
# headless's own minimal runtime dependencies on a bare Debian slim image
# (confirmed needed - a bare slim image is missing these and cv2 fails to
# import without them, even in headless mode).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       poppler-utils \
       libglib2.0-0 \
       libsm6 \
       libxext6 \
       libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Azure App Service for Containers / Container Apps both inject the
# listening port via $PORT (App Service defaults it to 8000 if unset;
# Container Apps requires it to match the configured target port - set
# that to 8000 in the Container App config to match this default).
ENV PORT=8000
EXPOSE 8000

# No --reload (that's dev-only - see app/main.py's own reload-detection
# logging). $HOME/data is NOT durable on App Service's own filesystem by
# default (see the deployment notes) - configure Azure Blob Storage via
# the AZURE_STORAGE_* env vars for anything that must survive a restart.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
