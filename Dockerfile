FROM jinaai/jina:master as base

COPY . ./big_transfer/
WORKDIR ./big_transfer

RUN pip install .

FROM base
RUN pip install -r tests/requirements.txt
RUN pytest tests

FROM base
ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]