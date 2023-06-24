# Builder image
FROM pytorch/torchserve AS builder

WORKDIR /usr/app
ADD requirements.txt .
RUN pip install -r requirements.txt

ADD dump_model.py .
RUN python dump_model.py

ADD handler.py .
ADD scripts/create-archive.sh scripts/create-archive.sh

RUN ./scripts/create-archive.sh

# Production image
FROM pytorch/torchserve

ADD requirements.txt .
RUN pip install -r requirements.txt

ADD docker/entrypoint.sh entrypoint.sh
ADD scripts/start-torchserve.sh start-torchserve.sh

COPY --from=builder /usr/app/model_store model_store

ENTRYPOINT [ "./entrypoint.sh" ]

CMD ["./start-torchserve.sh"]