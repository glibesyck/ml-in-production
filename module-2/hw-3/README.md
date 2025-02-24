# Deploying MinIO

## 1. Deploy MinIO Locally

I provide instructions for MacOS only as it is the OS I can check the steps to be working. One can find official docs for Linux [here](https://min.io/docs/minio/linux/operations/install-deploy-manage/deploy-minio-single-node-single-drive.html).
For MacOS, I will basically follow this [guide](https://min.io/docs/minio/macos/index.html).

### Steps
1. Download MinIO Server from `brew`:
   ```sh
   brew install minio/stable/minio
   ```
2. Launch the MinIO Server:

Choose appropriate DIR you want to use.
   ```sh
   export MINIO_CONFIG_ENV_FILE=/etc/default/minio
   minio server DIR --console-address :9001
   ```
   In my case, I'm located in `module-2` folder and choose DIR to be `hw-3`.

3. Access the MinIO web UI at [http://localhost:9001](http://localhost:9001). Default credentials:
   - **Username:** `minioadmin`
   - **Password:** `minioadmin`

## 2. Deploy MinIO with Docker

### Steps
1. Pull the MinIO Docker image:
   ```sh
   docker pull minio/minio
   ```
2. Run MinIO in a container:
   ```sh
   docker run -p 9000:9000 -p 9090:9090 --name minio \
     -e MINIO_ROOT_USER=minioadmin \
     -e MINIO_ROOT_PASSWORD=minioadmin \
     -v ~/minio-data:/data \
     minio/minio server /hw-3 --console-address ":9090"
   ```
3. Access the MinIO console at [http://localhost:9090](http://localhost:9090).

## 3. Deploy MinIO on Kubernetes

### Steps
1. Create Persistent Volume Claim:

```sh
kubectl create -f https://raw.githubusercontent.com/kubernetes/examples/master/staging/storage/minio/minio-standalone-pvc.yaml
```

2. Create Minio Deployment:
   ```sh
   kubectl create -f https://raw.githubusercontent.com/kubernetes/examples/master/staging/storage/minio/minio-standalone-deployment.yaml
   ```
3. Create Minio Service:
   ```sh
   kubectl create -f https://raw.githubusercontent.com/kubernetes/examples/master/staging/storage/minio/minio-standalone-service.yaml
   ```

