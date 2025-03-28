from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from app.modnet_model import load_model, remove_background
import os
import asyncio
from typing import List
import base64

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Allow React development server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once at the startup
model_ckpt_path = os.path.join(os.path.dirname(__file__), 'modnet_photographic_portrait_matting.ckpt')
model = load_model(model_ckpt_path)

async def process_single_image(file: UploadFile, model):
    try:
        # Read the uploaded image
        image_bytes = await file.read()
        
        # Process the image with the model
        result_image = remove_background(image_bytes, model)
        
        # Convert the result to base64
        img_byte_arr = BytesIO()
        result_image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        
        return {
            "filename": file.filename,
            "status": "success",
            "image": img_base64
        }
    except Exception as e:
        return {
            "filename": file.filename,
            "status": "error",
            "error": str(e)
        }

@app.post("/remove_background/")
async def remove_bg(file: UploadFile = File(...)):
    # Log file details
    print(f"Received file: {file.filename}")
    # Read the uploaded image
    image_bytes = await file.read()

    # Process the image with the model
    result_image = remove_background(image_bytes, model)

    # Save the result to a BytesIO object and send as response
    img_byte_arr = BytesIO()
    result_image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")

@app.post("/remove_background_multiple/")
async def remove_bg_multiple(files: List[UploadFile] = File(...)):
    # Log number of files received
    print(f"Received {len(files)} files")
    
    # Process all images in parallel
    tasks = [process_single_image(file, model) for file in files]
    results = await asyncio.gather(*tasks)
    
    return JSONResponse(content={
        "status": "success",
        "results": results
    })
