# CreditCardFraudDetectionSystem

#### Requirements:

+ Ubuntu - 20.04
+ python - 3.11

#### Dataset Preparation

1. Install kagglehub
2. Download data into **Dataset** folder

```bash
import kagglehub

# Download latest version
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

print("Path to dataset files:", path)
```

#### Instructions to follow:

1. Create and activate conda environment

```bash
conda create --name env_name python==3.11 -y
conda activate env_name
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Start the application
```bash
python main.py
```

#### Instructions for Docker

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

#### To train the Model

1. Update .env file and run the below script

2. Start the MLflow Server

```bash
source start_mlflow.sh
```
**Note:** Cross-chek **MLflow server** address in both **.env** & **start_mlflow.sh**

3. Start training

```bash
python train.py
```
**Note:** Trained Models will be saved into Weights folder.


