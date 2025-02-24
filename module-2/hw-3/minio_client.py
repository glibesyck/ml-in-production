from minio import Minio
from minio.error import S3Error


class MinioClient:
    def __init__(self, endpoint, access_key, secret_key, secure):
        self.client = Minio(endpoint, access_key = access_key, secret_key =  secret_key, secure = secure)

    def create_bucket(self, bucket_name):
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                print(f"Bucket '{bucket_name}' created.")
            else:
                print(f"Bucket '{bucket_name}' already exists.")
        except S3Error as e:
            print(f"Error creating bucket {bucket_name}: {e}")

    def upload_file(self, bucket_name, file_path, object_name):
        try:
            self.client.fput_object(bucket_name, object_name, file_path)
            print(f"File '{object_name}' uploaded to bucket '{bucket_name}'.")
        except S3Error as e:
            print(f"Error uploading file {object_name}: {e}")

    def download_file(self, bucket_name, object_name, file_path):
        try:
            self.client.fget_object(bucket_name, object_name, file_path)
            print(f"File '{object_name}' downloaded to '{file_path}'.")
        except S3Error as e:
            print(f"Error downloading file {object_name}: {e}")

    def delete_file(self, bucket_name, object_name):
        try:
            self.client.remove_object(bucket_name, object_name)
            print(f"File '{object_name}' deleted from bucket '{bucket_name}'.")
        except S3Error as e:
            print(f"Error deleting file {object_name}: {e}")
