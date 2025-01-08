# CreditCardFraudDetectionSystem

## Requirements:

+ Ubuntu - 20.04
+ python - 3.11


### Instructions to follow:

1. Create and activate conda environment

```
conda create --name env_name python==3.11
conda activate env_name
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. start the application
```
python demo.py
```

### Instructions for Docker

1. Build the docker image

```
docker compose build
```

2. Start the application

```
docker comose -d up
```

**Note**: Delete the container after usage

```
docker compose down
```