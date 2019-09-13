
FROM python:3.7.4-slim

WORKDIR /app

COPY . /app

RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 100

CMD ["python", "real1.py"]
