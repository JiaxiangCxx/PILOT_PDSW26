#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CONDA=${CONDA:-conda}
ENV_PREFIX=${PILOT_ENV_PREFIX:-$ROOT/.conda/pilot-ae}

command -v "$CONDA" >/dev/null 2>&1||{
  echo "Conda was not found. Install Miniconda or set CONDA=/path/to/conda." >&2
  exit 1
}

if [[ -d $ENV_PREFIX ]];then
  echo "PILOT environment already exists: $ENV_PREFIX"
else
  "$CONDA" env create --prefix "$ENV_PREFIX" --file "$ROOT/artifact/environment.yml"
fi

cat <<EOF

Activate the project-local environment with:
  conda activate "$ENV_PREFIX"

Then validate the artifact with:
  python artifact/check_artifact.py
EOF
