import kaggle

kaggle.api.authenticate()

# Download latest version

kaggle.api.dataset_download_files('The_name_of_the_dataset', path='Dataset', unzip=True)
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

print("Path to dataset files:", path)