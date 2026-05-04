#!/usr/bin/env bash
set -euo pipefail

# ===========================================
# Configuration (modifiable)
# ===========================================
PROJECT_ID="${PROJECT_ID:-ton-project-id}"
REGION="${REGION:-us-central1}"
BUCKET="${BUCKET:-ton-bucket-vertex-unique-global}"   # Nom global unique
GCS_PREFIX="${GCS_PREFIX:-songo-stockfish}"
SA_NAME="${SA_NAME:-songo-vertex-runtime}"

if [[ "${PROJECT_ID}" == "ton-project-id" ]]; then
  echo "Erreur: renseigne PROJECT_ID avant execution."
  echo "Exemple:"
  echo "PROJECT_ID='mon-projet-gcp' BUCKET='mon-bucket-vertex-unique' ./setup_vertex.sh"
  exit 1
fi

if [[ "${BUCKET}" == "ton-bucket-vertex-unique-global" ]]; then
  echo "Erreur: renseigne BUCKET avant execution."
  echo "Exemple:"
  echo "PROJECT_ID='mon-projet-gcp' BUCKET='mon-bucket-vertex-unique' ./setup_vertex.sh"
  exit 1
fi

echo "==> Auth gcloud (si deja connecte, la commande reutilise la session)"
gcloud auth login --update-adc
gcloud config set project "${PROJECT_ID}"
gcloud config set ai/region "${REGION}"

SUBMITTER_ACCOUNT="$(gcloud config get-value account 2>/dev/null)"
if [[ "${SUBMITTER_ACCOUNT}" == *"gserviceaccount.com" ]]; then
  SUBMITTER_MEMBER="serviceAccount:${SUBMITTER_ACCOUNT}"
else
  SUBMITTER_MEMBER="user:${SUBMITTER_ACCOUNT}"
fi

PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "PROJECT_ID=${PROJECT_ID}"
echo "PROJECT_NUMBER=${PROJECT_NUMBER}"
echo "REGION=${REGION}"
echo "BUCKET=gs://${BUCKET}"
echo "SUBMITTER=${SUBMITTER_MEMBER}"
echo "RUNTIME_SA=${SA_EMAIL}"

echo "==> Activation des APIs requises"
gcloud services enable \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  iam.googleapis.com \
  serviceusage.googleapis.com \
  cloudresourcemanager.googleapis.com \
  compute.googleapis.com \
  --project "${PROJECT_ID}"

echo "==> Creation du service account runtime (si absent)"
if ! gcloud iam service-accounts describe "${SA_EMAIL}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud iam service-accounts create "${SA_NAME}" \
    --project "${PROJECT_ID}" \
    --display-name "Songo Vertex Runtime SA"
fi

echo "==> Creation du bucket (si absent)"
if ! gcloud storage buckets describe "gs://${BUCKET}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://${BUCKET}" \
    --project "${PROJECT_ID}" \
    --location "${REGION}" \
    --uniform-bucket-level-access
fi

echo "==> Roles IAM (submission + attachement SA + acces GCS)"
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="${SUBMITTER_MEMBER}" \
  --role="roles/aiplatform.user"

gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
  --member="${SUBMITTER_MEMBER}" \
  --role="roles/iam.serviceAccountUser"

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.objectUser"

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="${SUBMITTER_MEMBER}" \
  --role="roles/storage.objectUser"

echo "==> Ecriture des variables pour le notebook"
cat > .vertex_env.sh <<EOF
export SONGO_VERTEX_PROJECT_ID="${PROJECT_ID}"
export SONGO_VERTEX_REGION="${REGION}"
export SONGO_VERTEX_GCS_BUCKET="${BUCKET}"
export SONGO_VERTEX_GCS_PREFIX="${GCS_PREFIX}"
export SONGO_VERTEX_SERVICE_ACCOUNT="${SA_EMAIL}"
EOF

echo
echo "Setup termine."
echo "Fichier cree: $(pwd)/.vertex_env.sh"
echo "Charge-le avec: source .vertex_env.sh"
