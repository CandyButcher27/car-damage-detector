# Kubernetes manifests

Single-replica, stateless deployment for the `data-ingestion` service.

## Apply order

```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml      # includes ServiceAccount + initContainer
kubectl apply -f k8s/service.yaml
```

Or in one shot:

```bash
kubectl apply -f k8s/
```

## Topology

| File             | Purpose                                                   |
| ---------------- | --------------------------------------------------------- |
| `configmap.yaml` | All env config (CORS, limits, model paths, GCS bucket)    |
| `deployment.yaml`| Deployment + ServiceAccount + GCS init container         |
| `service.yaml`   | ClusterIP service on port 80 → pod port 8000              |

**No HPA**, **no PVC**, **no PDB** — by design. The user's cluster is resource
constrained and runs other spike-prone workloads. We trade redundancy for
predictability and cost.

## Models from GCS

The `model-sync` init container pulls model artefacts from a GCS bucket into
an in-pod `emptyDir`. On pod restart the models are re-pulled (~10-30 s on
GCP internal network).

Set the bucket URI in the ConfigMap:

```yaml
UPSURE_MODELS_GCS_URI: "gs://upsure-ai-models/data-ingestion/v1.1.0"
```

Authentication options (pick one):

1. **Workload Identity (recommended)** — annotate the ServiceAccount with
   `iam.gke.io/gcp-service-account` and grant
   `roles/storage.objectViewer` on the bucket. No Kubernetes Secret needed.

2. **Service-account key file** — create a Secret named
   `data-ingestion-gcs-key` containing `key.json`, then uncomment the
   `gcs-key` volume and mount block in `deployment.yaml`.

If `UPSURE_MODELS_GCS_URI` is empty, the init container exits cleanly and
the API container expects models to be present in the `models` emptyDir
(useful in CI or for `docker compose` testing where you bind-mount).

## Probes

| Probe       | Path        | Effect of failure                       |
| ----------- | ----------- | --------------------------------------- |
| `startup`   | `/livez`    | Pod stays "Starting" up to 5 minutes    |
| `liveness`  | `/livez`    | Pod restarted after 3 consecutive       |
| `readiness` | `/readyz`   | Pod removed from Service endpoints      |

`/readyz` only returns 200 once every *critical* model is loaded. Set
`UPSURE_REQUIRE_MODELS_FOR_READY=false` in the ConfigMap if you want to
ship traffic to a pod whose damage model is still warming up.

## Metrics

Pods expose Prometheus metrics on `/metrics`. The
`prometheus.io/*` pod annotations are picked up by the cluster's
Prometheus operator out of the box.

## CORS

Set the comma-separated allowlist in the ConfigMap:

```yaml
UPSURE_CORS_ORIGINS: "https://digi-motor.upsure.io,https://app.upsure.io"
UPSURE_CORS_ALLOW_CREDENTIALS: "false"
```

## Rolling a new image

```bash
kubectl set image deploy/data-ingestion api=registry.upsure.io/ai-cohort/data-ingestion:1.2.0
kubectl rollout status deploy/data-ingestion
```

`strategy: Recreate` means the old pod is killed before the new one comes
up — there will be a brief window (typically 30-60 s including model sync
and readiness) where the service is unavailable. The envelope's
`retryable: true` on 5xx errors signals the UI to retry.
