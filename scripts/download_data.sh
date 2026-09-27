#!/usr/bin/env bash
#
# Fetch the H&M Personalized Fashion Recommendations dataset.
#
#   ./scripts/download_data.sh            CSVs only  (~3.5 GB)
#   ./scripts/download_data.sh --images   CSVs + article images (~30 GB)
#
# Requires a Kaggle account, acceptance of the competition rules, and a logged-in
# kaggle CLI (`kaggle auth login`). See the README.
set -euo pipefail
cd "$(dirname "$0")/.."

COMPETITION="h-and-m-personalized-fashion-recommendations"
RAW="data/raw"
WANT_IMAGES="${1:-}"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "error: the kaggle CLI is not installed." >&2
  echo "  pip install kaggle" >&2
  exit 1
fi

# Probe the API rather than looking for a credentials file: the CLI accepts an
# OAuth login (`kaggle auth login`), ~/.kaggle/access_token, KAGGLE_API_TOKEN,
# or the legacy ~/.kaggle/kaggle.json. The probe also catches unaccepted rules.
if ! probe="$(kaggle competitions files -c "$COMPETITION" 2>&1)"; then
  if grep -qi "authentication required" <<<"$probe"; then
    echo "error: not authenticated with Kaggle. Either:" >&2
    echo "  kaggle auth login                  (browser login, recommended)" >&2
    echo "  or save a token from https://www.kaggle.com/settings/api to ~/.kaggle/access_token" >&2
  elif grep -qiE "403|forbidden|rules" <<<"$probe"; then
    echo "error: Kaggle refused access. Accept the competition rules first:" >&2
    echo "  https://www.kaggle.com/competitions/$COMPETITION/rules" >&2
  else
    echo "error: Kaggle API check failed:" >&2
    echo "$probe" >&2
  fi
  exit 1
fi

mkdir -p "$RAW"

# The CSVs are small enough to pull individually, which avoids downloading the
# 30 GB image archive just to get at the transaction table.
for f in transactions_train.csv customers.csv articles.csv sample_submission.csv; do
  if [[ -f "$RAW/$f" ]]; then
    echo "have    $f"
    continue
  fi
  echo "fetching $f ..."
  kaggle competitions download -c "$COMPETITION" -f "$f" -p "$RAW"
  [[ -f "$RAW/$f.zip" ]] && unzip -q -o "$RAW/$f.zip" -d "$RAW" && rm "$RAW/$f.zip"
done

if [[ "$WANT_IMAGES" == "--images" ]]; then
  echo
  echo "Fetching images (~30 GB). This takes a while."
  kaggle competitions download -c "$COMPETITION" -f images -p "$RAW" || {
    echo "Per-file image download unavailable; falling back to the full archive." >&2
    kaggle competitions download -c "$COMPETITION" -p "$RAW"
  }
  for z in "$RAW"/*.zip; do
    [[ -f "$z" ]] && unzip -q -o "$z" -d "$RAW" && rm "$z"
  done
fi

echo
echo "Contents of $RAW:"
ls -lh "$RAW" | tail -n +2 | awk '{printf "  %-34s %s\n", $9, $5}'
echo
echo "Next: python -m ensemble.data.ingest"
