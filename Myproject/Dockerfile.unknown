# ==========================================
#  Dockerfile for Trading Project
#  Python 3.11.9 + Flask
# ==========================================

# ---- Base image ----
FROM python:3.11-slim

# ---- Set working directory ----
WORKDIR /app

# ---- Copy requirements first and install dependencies ----
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- Copy all project files ----
COPY . .

# ---- Expose Flask default port ----
EXPOSE 5000

# ---- Run the Flask app ----
CMD ["python", "app.py"]
