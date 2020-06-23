FROM python:3.8.3-slim-buster

COPY ./models /app/models
COPY ./configs.py /app/
COPY ./experimental_train.py /app/
COPY ./main.py /app/
COPY ./prepare_database.py /app/
COPY ./results.ipynb /app/
COPY ./train.py /app/
COPY ./utils.py /app/

WORKDIR /app

RUN pip install tensorflow \
        && pip install numpy \
        && pip install scikit-learn \
        && pip install matplotlib \
        && pip install notebook

ENTRYPOINT [ "python", "./main.py" ]