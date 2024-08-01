from fastapi import FastAPI, File, UploadFile
import boto3
from botocore.exceptions import NoCredentialsError
import os

app = FastAPI(title="SAMSUL API", version="1.0.0")

AWS_S3_BUCKET = os.getenv("LOADER_S3_BUCKET", "default-bucket")
AWS_ACCESS_KEY_ID = os.getenv("LOADER_AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("LOADER_AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("LOADER_AWS_REGION", "us-east-1")
AWS_S3_ENDPOINT_URL = os.getenv("LOADER_ENDPOINT_URL")
USE_SSL = os.getenv("LOADER_USE_SSL", "True").lower() == "true"

s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    endpoint_url=AWS_S3_ENDPOINT_URL,
    use_ssl=USE_SSL
)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        s3_client.put_object(
            Bucket=AWS_S3_BUCKET,
            Key=file.filename,
            Body=file_content,
            ContentType=file.content_type,
        )
        return {"message": "File uploaded successfully", "filename": file.filename}
    except NoCredentialsError:
        return {"error": "Credentials not available"}
    except Exception as e:
        return {"error": str(e)}
