FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y python3-pip build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 --version

WORKDIR /workspace

COPY weights/RandomForestClassifier.pkl /workspace/weights/RandomForestClassifier.pkl

COPY main.py /workspace

COPY requirements.txt /workspace

RUN pip3 install -r requirements.txt

# Expose port and set entrypoint
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"]