docker run -p 9000:9000 -p 9090:9090 --name minio \
     -e MINIO_ROOT_USER=minioadmin \
     -e MINIO_ROOT_PASSWORD=minioadmin \
     -v ~/minio-data:/data \
     minio/minio server /hw-3 --console-address ":9090"

pytest test_minio_client.py