FROM jinaai/jina:2.0

COPY . /big_transfer/
WORKDIR /big_transfer

RUN pip install .

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]