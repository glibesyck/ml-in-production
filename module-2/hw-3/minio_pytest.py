import os
import tempfile
import pytest
from minio_client import MinioClient


@pytest.fixture
def minio_client():
    client = MinioClient("localhost:9000", access_key = "minioadmin", secret_key = "minioadmin", secure = False)
    return client


@pytest.fixture
def create_temp_file():
    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    tmp_file.write(b"This is a test file.")
    tmp_file.close()
    yield tmp_file.name
    os.remove(tmp_file.name)


def test_create_bucket(minio_client):
    bucket_name = "test-bucket"
    minio_client.create_bucket(bucket_name)

    assert minio_client.client.bucket_exists(bucket_name)


def test_upload_file(minio_client, create_temp_file):
    bucket_name = "test-bucket"
    object_name = "test-file.txt"

    minio_client.upload_file(bucket_name, create_temp_file, object_name)

    objects = minio_client.client.list_objects(bucket_name)
    object_names = [obj.object_name for obj in objects]
    assert object_name in object_names


def test_download_file(minio_client):
    bucket_name = "test-bucket"
    object_name = "test-file.txt"
    download_path = tempfile.NamedTemporaryFile(delete=False).name

    minio_client.download_file(bucket_name, object_name, download_path)

    with open(download_path, 'rb') as f:
        content = f.read()
        assert content == b"This is a test file."

    os.remove(download_path)


def test_delete_file(minio_client):
    bucket_name = "test-bucket"
    object_name = "test-file.txt"

    minio_client.delete_file(bucket_name, object_name)

    objects = minio_client.client.list_objects(bucket_name)
    object_names = [obj.object_name for obj in objects]
    assert object_name not in object_names
