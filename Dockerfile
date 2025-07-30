FROM python:3.10
WORKDIR /app

COPY requirements.txt requirements.txt
COPY app.py app.py
COPY linear_model.pkl linear_model.pkl
COPY preprocessing.pkl preprocessing.pkl 
COPY data/processed/dataset_cleaned.csv data/processed/dataset_cleaned.csv
COPY templates/ templates/

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "app.py"]
