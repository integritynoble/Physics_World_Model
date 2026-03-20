# Hosted Infrastructure Planning

## Architecture Overview

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Web UI     │────▶│   API Server │────▶│  Job Queue   │
│  (React/Next)│     │  (FastAPI)   │     │  (Celery)    │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                           ┌──────────────────────┤
                           ▼                      ▼
                     ┌──────────┐           ┌──────────┐
                     │ GPU Worker│           │ GPU Worker│
                     │ (PWM Core)│           │ (PWM Core)│
                     └──────────┘           └──────────┘
                           │                      │
                           ▼                      ▼
                     ┌────────────────────────────────┐
                     │     Object Storage (S3/GCS)    │
                     │   RunBundles + Datasets + Cache │
                     └────────────────────────────────┘
```

## Components

### 1. API Server
- **Framework**: FastAPI
- **Auth**: API keys + OAuth2 for web UI
- **Endpoints**:
  - `POST /jobs` — Submit calibration/reconstruction job
  - `GET /jobs/{id}` — Job status + results
  - `GET /jobs/{id}/runbundle` — Download RunBundle
  - `POST /datasets` — Upload measurement data
  - `GET /health` — Service health check

### 2. Job Queue
- **Backend**: Celery + Redis
- **Job types**: calibrate, reconstruct, calibrate_and_reconstruct, generate_dataset
- **Priority levels**: free (low), paid (high), sprint (highest)
- **Timeout**: 1 hour default, 4 hours for sprints

### 3. GPU Workers
- **Hardware**: NVIDIA A100 or T4 instances
- **Scaling**: Auto-scale 0-4 workers based on queue depth
- **Image**: Docker container with pwm_core + CUDA + PyTorch
- **Isolation**: Each job runs in isolated container

### 4. Storage
- **RunBundles**: Content-addressed storage (SHA256 keys)
- **Datasets**: User-scoped buckets with encryption at rest
- **Cache**: Shared compiled graph cache across workers
- **Retention**: 30 days free tier, unlimited for paid

### 5. Monitoring
- **Metrics**: Prometheus + Grafana
- **Logging**: Structured JSON logs → ELK stack
- **Alerts**: Job failures, queue depth, GPU utilization

## Cost Model

| Resource | Unit Cost | Free Tier | Paid Tier |
|----------|-----------|-----------|-----------|
| GPU compute | $0.50/GPU-min | 10 min/month | Pay-as-you-go |
| Storage | $0.02/GB/month | 1 GB | 100 GB |
| API calls | $0.001/call | 100/day | Unlimited |

## Deployment Phases

### Phase 1: MVP (Month 1)
- Single GPU worker on GCP
- FastAPI server + Redis queue
- Basic auth (API keys)
- GCS storage

### Phase 2: Scale (Month 2-3)
- Auto-scaling GPU pool
- Web UI for job submission
- OAuth2 authentication
- Usage tracking + billing

### Phase 3: Enterprise (Month 4+)
- Multi-region deployment
- SLA guarantees
- Custom VPC deployment
- Dedicated GPU instances
