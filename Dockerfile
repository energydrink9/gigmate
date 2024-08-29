FROM python:3.12

WORKDIR /app

COPY . /app

RUN python -m pip install .

RUN echo 'api { credentials {"access_key": "<KEY>", "secret_key": "<KEY>"} }' > ~/clearml.conf

CMD ["python", "-m", "gigmate.train"]