# Machine Learning - voice recognition

Repository contains minimal amount of code to provide UI for voice recognition model.

## Build & run locally with Docker (recommended)

Build an image:

```docker
docker build -t iml-app .
```

Run app:

```docker
docker run -p 8080:8080 iml-app
```

Then app is up and running on `localhost:8080`.

## Or run locally from source code

1. Create virtual environment, activate it and run:

```
pip install -r requirements.txt
```
2. Then, run it via streamlit:
```bash
streamlit run <path/to/app.py>
```

## App's preview

https://github.com/user-attachments/assets/72a94816-52a4-494f-a378-d4b96a70d509
