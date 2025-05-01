1. Build docker image

`docker build -t streamlit-app:latest .`

2. Copy image to minikube

`minikube image load streamlit-app:latest`

3. Apply Kubernetes manifest

`kubectl apply -f deployment.yaml`