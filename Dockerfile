FROM amd64/python:3.10-slim

RUN mkdir /home/src

COPY ./docker/main.py /home/
COPY ./docker/requirements.txt /home/
COPY ./docker/server.py /home/
COPY ./docker/model.py /home/
COPY ./src/models.py /home/src/
COPY ./src/pipeline.py /home/src/
COPY ./src/preprocess.py /home/src/
COPY ./src/utils.py /home/src/

WORKDIR /home/

RUN python3 -m pip install --upgrade pip && pip install -r requirements.txt

ENTRYPOINT ["python","-u", "./main.py"]

CMD ["python", "-u", "./main.py"]