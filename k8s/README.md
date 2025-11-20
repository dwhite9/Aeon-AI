# Aeon Kubernetes Manifests

This directory contains all Kubernetes deployment configurations for the Aeon platform.

## Directory Structure

```
k8s/
├── base/           # Infrastructure services (databases, monitoring)
│   ├── namespace.yaml
│   ├── redis.yaml
│   ├── postgres.yaml
│   ├── qdrant.yaml
│   └── monitoring.yaml
├── app/            # Application deployments
│   ├── api-backend.yaml
│   └── ui-frontend.yaml
└── jobs/           # CronJobs for background tasks
    └── nightly-optimize.yaml
```

## Prerequisites

1. **K3s installed** (or any Kubernetes cluster)
   ```bash
   curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--disable traefik" sh -
   export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
   ```

2. **Helm installed**
   ```bash
   curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
   ```

3. **NGINX Ingress Controller**
   ```bash
   kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/baremetal/deploy.yaml
   ```

4. **Local Docker Registry** (for development)
   ```bash
   docker run -d -p 5000:5000 --restart=always --name registry registry:2
   ```

## Deployment Order

### 1. Add Helm Repositories

```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
```

### 2. Deploy Infrastructure (base/)

```bash
# Create namespace
kubectl apply -f base/namespace.yaml

# Deploy Qdrant vector database
kubectl apply -f base/qdrant.yaml

# Deploy Redis (using Helm)
helm install redis bitnami/redis \
  --set auth.enabled=false \
  --set master.persistence.enabled=false \
  --set master.resources.requests.memory=4Gi \
  --set master.resources.requests.cpu=1

# Deploy PostgreSQL (using Helm)
helm install postgres bitnami/postgresql \
  --set primary.persistence.size=20Gi \
  --set primary.persistence.storageClass=local-path \
  --set primary.resources.requests.memory=8Gi \
  --set primary.resources.requests.cpu=2 \
  --set auth.database=aiplatform \
  --set auth.username=aiuser \
  --set auth.password=changeme

# Deploy Prometheus + Grafana (using Helm)
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=20Gi \
  --set prometheus.prometheusSpec.resources.requests.memory=4Gi \
  --set grafana.service.type=NodePort \
  --set grafana.service.nodePort=30001
```

### 3. Build and Push Application Images

**Backend:**
```bash
cd services
docker build -t localhost:5000/aeon-api:latest .
docker push localhost:5000/aeon-api:latest
```

**Frontend:**
```bash
cd ui
docker build -t localhost:5000/aeon-ui:latest .
docker push localhost:5000/aeon-ui:latest
```

### 4. Deploy Application (app/)

**IMPORTANT**: Update `api-backend.yaml` with your host IP address before deploying:
```yaml
VLLM_ENDPOINT: "http://YOUR_HOST_IP:8000/v1"
EMBEDDING_ENDPOINT: "http://YOUR_HOST_IP:8001"
```

```bash
# Deploy backend
kubectl apply -f app/api-backend.yaml

# Deploy frontend
kubectl apply -f app/ui-frontend.yaml

# Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app=api-backend --timeout=300s
kubectl wait --for=condition=ready pod -l app=ui-frontend --timeout=300s
```

### 5. Deploy Background Jobs (jobs/)

```bash
kubectl apply -f jobs/nightly-optimize.yaml
```

## Accessing Services

### Option 1: Using Ingress

1. Add to `/etc/hosts`:
   ```
   127.0.0.1 aeon.local
   ```

2. Access at: http://aeon.local

### Option 2: Using Port Forwarding

```bash
# Frontend
kubectl port-forward svc/ui-frontend 3000:80

# Backend API
kubectl port-forward svc/api-backend 8080:8080

# Grafana
# Already exposed on NodePort 30001
# Access at: http://localhost:30001
# Default credentials: admin/prom-operator
```

## Monitoring

### Check Pod Status

```bash
# All pods
kubectl get pods -A

# Aeon pods
kubectl get pods -l component=backend
kubectl get pods -l component=frontend

# Vector DB
kubectl get pods -n vector-db
```

### View Logs

```bash
# Backend logs
kubectl logs -f deployment/api-backend

# Frontend logs
kubectl logs -f deployment/ui-frontend

# Qdrant logs
kubectl logs -f -n vector-db statefulset/qdrant
```

### Resource Usage

```bash
# Overall cluster usage
kubectl top nodes
kubectl top pods

# Specific components
kubectl top pod -l app=api-backend
```

## Configuration

### Update Environment Variables

Edit `app/api-backend.yaml` ConfigMap:
```yaml
data:
  VLLM_ENDPOINT: "http://YOUR_HOST_IP:8000/v1"
  EMBEDDING_ENDPOINT: "http://YOUR_HOST_IP:8001"
```

Then apply changes:
```bash
kubectl apply -f app/api-backend.yaml
kubectl rollout restart deployment/api-backend
```

### Update Secrets

```bash
# Create new secret
kubectl create secret generic api-backend-secrets \
  --from-literal=POSTGRES_PASSWORD=new_password \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart deployment to pick up changes
kubectl rollout restart deployment/api-backend
```

## Scaling

```bash
# Scale backend
kubectl scale deployment api-backend --replicas=3

# Scale frontend
kubectl scale deployment ui-frontend --replicas=3
```

## Troubleshooting

### Pods Not Starting

```bash
# Describe pod for events
kubectl describe pod POD_NAME

# Check logs
kubectl logs POD_NAME

# Check resource constraints
kubectl top pods
```

### Connection Issues

```bash
# Test backend from within cluster
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- \
  curl http://api-backend:8080/health

# Test Qdrant
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- \
  curl http://qdrant.vector-db:6333
```

### Image Pull Errors

```bash
# Check if local registry is running
docker ps | grep registry

# Re-push images
docker push localhost:5000/aeon-api:latest
docker push localhost:5000/aeon-ui:latest
```

## Cleanup

```bash
# Delete applications
kubectl delete -f app/
kubectl delete -f jobs/

# Delete infrastructure
kubectl delete -f base/
helm uninstall redis
helm uninstall postgres
helm uninstall kube-prometheus-stack

# Delete K3s (complete cleanup)
/usr/local/bin/k3s-uninstall.sh
```
