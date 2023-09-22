FROM python:3.9
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY app /app/
COPY model app/model/
COPY utils app/utils/
WORKDIR /app

CMD ["uvicorn", "main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]
EXPOSE 80