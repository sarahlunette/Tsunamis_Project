FROM python:3.11

WORKDIR /apps

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./apps /apps

EXPOSE 8000

CMD ["uvicorn", "apps.main:api", "--host", "0.0.0.0", "--port", "8000"]