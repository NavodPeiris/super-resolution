You have to download the models for super resolution and colorization

download model for super resolution by this url: https://drive.google.com/file/d/1TPrz5QKd8DHHt1k8SRtm6tMiPjz_Qene/view?usp=drive_link

then place the model **RRDB_ESRGAN_x4.pth** inside superResolution/src/models folder

**pip install -r requirements.txt**
      - this will install the required packages

**cd src**
**uvicorn test:app --host 0.0.0.0 --reload**
    - this will start server