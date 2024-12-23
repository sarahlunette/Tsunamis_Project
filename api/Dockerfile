FROM python:3.11

WORKDIR /apps 

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./apps/main.py main.py

EXPOSE 8000

CMD ["uvicorn", "main:api", "--host", "0.0.0.0", "--port", "8000"]