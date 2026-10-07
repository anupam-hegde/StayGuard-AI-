# 🏨 Hotel Reservation Cancellation Prediction – End-to-End MLOps Pipeline

An **end-to-end MLOps pipeline** for a Hotel Reservation prediction system demonstrating how machine learning workflows can be **automated, containerized, and deployed** using modern **DevOps and cloud-native tools**.

This project showcases **production-style ML infrastructure**, including model training, experiment tracking, CI/CD automation, Kubernetes deployment, and monitoring.

---

## 🚀 Tech Stack

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Orchestration-blue)
![Jenkins](https://img.shields.io/badge/Jenkins-CI/CD-red)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue)
![Kubeflow](https://img.shields.io/badge/Kubeflow-ML%20Pipelines-orange)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-orange)
![Grafana](https://img.shields.io/badge/Grafana-Dashboards-orange)
![GCP](https://img.shields.io/badge/GCP-Cloud%20Platform-yellow)
![License](https://img.shields.io/badge/License-GPL--3.0-green)

---

# 📌 Project Overview

The goal of this project is to demonstrate how a **machine learning model moves through a complete production lifecycle**:

```
Development → Training → Containerization → CI/CD → Kubernetes Deployment → Monitoring
```

This pipeline integrates key **industry MLOps tools** to create a **scalable, automated, and reproducible ML workflow**.

---

# 🏗 System Architecture

The project follows a **modern MLOps architecture** where ML pipelines are automated and deployed using container orchestration.

### Workflow

1. Developer pushes code to **GitHub**
2. **Jenkins CI/CD pipeline** is triggered
3. **Docker image** is built and pushed to **Google Container Registry**
4. **Kubernetes (Minikube)** deploys the containerized application
5. **MLflow** tracks experiments and model versions
6. **Prometheus** collects metrics
7. **Grafana** visualizes monitoring dashboards

```
Developer
   │
   ▼
GitHub Repository
   │
   ▼
Jenkins CI/CD Pipeline
   │
   ├── Build Docker Image
   ├── Run ML Pipeline
   └── Push Image to GCR
   │
   ▼
Kubernetes Cluster (Minikube)
   │
   ├── Model Training
   ├── Model Serving
   └── Application Deployment
   │
   ▼
MLflow Tracking Server
   │
   ▼
Prometheus Metrics
   │
   ▼
Grafana Monitoring Dashboard
```

---

# 🤖 Machine Learning Pipeline

The ML workflow automates the full lifecycle of the model.

```
Raw Data
   │
   ▼
Data Ingestion
   │
   ▼
Data Preprocessing
   │
   ▼
Feature Engineering
   │
   ▼
Model Training
   │
   ▼
Model Evaluation
   │
   ▼
Experiment Tracking (MLflow)
   │
   ▼
Model Packaging (Docker)
   │
   ▼
Deployment (Kubernetes)
```

---

# ⚙ ML Pipeline Workflow

1️⃣ Data preprocessing
2️⃣ Model training
3️⃣ Experiment tracking with **MLflow**
4️⃣ Docker image creation
5️⃣ CI/CD pipeline execution via **Jenkins**
6️⃣ Image pushed to **Google Container Registry (GCR)**
7️⃣ Deployment on **Kubernetes (Minikube)**
8️⃣ Monitoring using **Prometheus + Grafana**

---

# 🔄 CI/CD Pipeline

The CI/CD pipeline automates the machine learning workflow.

### Pipeline Stages

* Pull latest code from repository
* Install dependencies
* Execute machine learning pipeline
* Build Docker image
* Push image to container registry
* Deploy application to Kubernetes

This ensures **continuous integration, automated builds, and reproducible deployments**.

---

# 📊 Monitoring and Observability

The deployed application is monitored using a **modern observability stack**.

### 🔎 Prometheus

Collects metrics from the application and Kubernetes cluster.

### 📈 Grafana

Visualizes metrics using **real-time dashboards** for system health and performance monitoring.

---

# 🧰 Skills Demonstrated

This project demonstrates practical experience with **production-grade MLOps systems**.

* End-to-End **MLOps Pipeline Design**
* **Machine Learning Workflow Automation**
* **CI/CD for ML Systems**
* **Containerization with Docker**
* **Kubernetes Deployment**
* **Experiment Tracking with MLflow**
* **Monitoring using Prometheus & Grafana**
* **Cloud-Native ML Infrastructure**

---

# 📜 License

This project is licensed under the **GPL-3.0 License**.
