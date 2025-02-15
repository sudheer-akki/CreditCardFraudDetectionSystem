# CreditCardFraudDetectionSystem

## Requirements:

+ Ubuntu - 20.04
+ python - 3.11


### Instructions to follow:

1. Create and activate conda environment

```bash
conda create --name env_name python==3.11 -y
conda activate env_name
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. start the application
```bash
python demo.py
```

### Instructions for Docker

1. Build the docker image

```bash
docker compose build --no-cache --progress=plain | tee build.log
```

2. Start the application

```bash
docker compose up
```

**Note**: Delete the container after usage

```bash
docker compose down
```