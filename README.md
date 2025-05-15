# CreditCardFraudDetectionSystem

#### Requirements:

+ Ubuntu - 20.04
+ python - 3.11

#### Dataset Preparation

1. Install kagglehub
2. Download data into **Dataset** folder

```sh
import kagglehub

# Download latest version
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

print("Path to dataset files:", path)
```

#### Instructions to follow:

1. Create and activate conda environment

```sh
conda create --name env_name python==3.11 -y
conda activate env_name
```

2. Install dependencies
``sh
pip install -r requirements.txt
```

3. Start the API 
```sh
python main.py
```
**Note:** Visit **http://127.0.0.1:8000** to access API.

4. Test API with data
```sh
python test_api.py
```

5. To test the model
```sh
python test.py
```

#### Instructions for Docker

1. Build the docker image

```sh
docker compose build --no-cache --progress=plain | tee build.log
```

2. Start the ÀPI using 

```sh
docker compose up
```

**Note**: Delete the container after usage

```sh
docker compose down
```

#### To train the Model

1. Update config file and run the below script

2. Start the MLflow Server

```sh
source start_mlflow.sh
```
**Note:** Cross-chek **MLflow server** address in both **.env** & **start_mlflow.sh**

3. Start training

```sh
python train.py
```
**Note:** Trained Models will be saved into Weights folder.

## Support

If you like this project, please consider supporting it with a ⭐!
