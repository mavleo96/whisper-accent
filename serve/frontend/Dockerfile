FROM python:3.12-slim

WORKDIR /app

COPY serve/frontend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY serve/frontend/app.py .

ENV BACKEND_URL="http://backend:8000"

CMD ["python", "app.py"]
