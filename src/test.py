import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

from fastapi import FastAPI, UploadFile, File
import shutil
import os
from fastapi.responses import FileResponse
import socket

app = FastAPI()

hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)
print("ip_address : ", ip_address)

@app.get('/superRes/health')
async def hi():
    return {"response": "server running"}

@app.post('/superRes/infer')
async def superRes(file: UploadFile = File(...)):

    source_folder = 'LR/*'
    out_folder = 'results/*'

    # deleting source images
    for path in glob.glob(source_folder):
        if os.path.exists(path):
            os.remove(path)

    # deleting result images
    for path in glob.glob(out_folder):
        if os.path.exists(path):
            os.remove(path)

    save_directory = "./LR"
    
    # Save the uploaded file
    file_path = os.path.join(save_directory, file.filename)
    with open(file_path, "wb") as image:
        shutil.copyfileobj(file.file, image)
        print("image written")

    model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
    device = "cuda"

    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    print('Model path {:s}. \nTesting...'.format(model_path))

    
    base = osp.splitext(osp.basename(file_path))[0]
    print(base)
    # read images
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    outpath = 'results/{:s}_rlt.png'.format(base)
    cv2.imwrite(outpath, output)

    return FileResponse(outpath)


