from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os
import shutil
from extract_feats_pose_api import extract_pose

app = FastAPI()

# Directory for temporary uploaded files
UPLOAD_DIR = "data_save"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/process")
async def process_video(file: UploadFile = File(...)):
    """
    Upload and process a video file
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.mp4'):
            raise HTTPException(status_code=400, detail="Only MP4 files are allowed")
        
        # Save uploaded file temporarily
        temp_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            # Process the video
            result_path = extract_pose(temp_path)
            
            if os.path.exists(result_path):
                return FileResponse(
                    result_path,
                    media_type="application/octet-stream",
                    filename=os.path.basename(result_path)
                )
            else:
                raise HTTPException(status_code=500, detail="Processing failed")
                
        finally:
            # Cleanup temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)