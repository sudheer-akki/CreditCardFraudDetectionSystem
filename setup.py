import setuptools

with open("README.md","r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "1.0.0"

REPO_NAME = "CreditCardFraudDetectionSystem"
AUTHOR_USER_NAME= "sudheer-akki"
SRC_REPO = "mlproject"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    description="A small python package for ML app",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://gitlab.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls = {
        "Bug Tracker": f"https://gitlab.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',

)