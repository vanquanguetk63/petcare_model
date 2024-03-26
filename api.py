from typing import Union
import os
from fastapi import FastAPI, UploadFile
import inference_on_image
from werkzeug.utils import secure_filename
from pathlib import Path
from fastapi.responses import JSONResponse
import datetime
import shutil

app = FastAPI()

UPLOAD_FOLDER = './data/images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

if not os.path.exists(UPLOAD_FOLDER):
    path = Path(UPLOAD_FOLDER)
    path.mkdir(parents=True)

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def upload_file(file: UploadFile):
    # file = requests.files['file']
    if file.filename == '':
        return JSONResponse(status_code=400, content={'description': 'No selected file'})
    if not allowed_file(file.filename):
        return JSONResponse(status_code=400, content={'description': 'Type image not allowed'})
    file_name = '.'.join(file.filename.split(".")[:-1])
    file_type = file.filename.split(".")[-1]
    filename = secure_filename(file_name) + "_" + str(int(datetime.datetime.now().timestamp() * 1000)) + "." + file_type
    path_image_local = os.path.join(UPLOAD_FOLDER, filename)

    try:
        with open(path_image_local, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()

    result = inference_on_image.inference(inference_on_image.read_image(path_image_local))
    return {'label': result}



# uvicorn api:app --host 0.0.0.0 --port 8000
# curl -X 'POST' 'http://0.0.0.0:8000/predict' -H 'accept: application/json' --form '20230320_103146.jpg"'