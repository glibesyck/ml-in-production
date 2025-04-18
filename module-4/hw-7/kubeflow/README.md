## Deploying Kubeflow pipelines using `minikube`

Before usage make sure you have Docker and minikube image being installed. [Reference article](https://medium.com/@vinayakshanawad/build-an-ml-pipeline-part-1-getting-started-with-kubeflow-v2-pipelines-74981b88db9a).

1. Start minikube in terminal:

`minikube start`

2. Enable Ingress addon:

`minikube addons enable ingress`

3. Apply manifest file from Kubeflow GitHub repo:

`kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=2.0.0"`

`kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io`

`kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=2.0.0"`

4. Check whether `kubeflow` is installed:

`kubectl get pods -A`

5. Run Kubeflow UI:

`kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80`

and access it:

`http://localhost:8080/`.