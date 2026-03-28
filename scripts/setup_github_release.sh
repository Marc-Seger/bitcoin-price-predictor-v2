#!/bin/bash
# One-time setup: upload master_df.csv to a GitHub release
# so the daily GitHub Action can download and update it.
#
# Usage: bash scripts/setup_github_release.sh

set -e

echo "Creating GitHub release 'latest' with master_df.csv..."

cd "$(dirname "$0")/.."

# Create the release and upload the file
gh release create latest data/full_data/master_df.csv \
  --title "Latest Data" \
  --notes "master_df.csv — updated daily by GitHub Actions.
Contains all historical data (prices, macro, sentiment, on-chain, technical indicators).
~2,300 rows, 280+ columns, Jan 2020 to present."

echo ""
echo "Done! Release created at:"
echo "https://github.com/Marc-Seger/bitcoin-price-predictor-v2/releases/tag/latest"
echo ""
echo "Next steps:"
echo "1. Go to https://github.com/Marc-Seger/bitcoin-price-predictor-v2/settings/secrets/actions"
echo "2. Click 'New repository secret'"
echo "3. Name: FRED_API_KEY"
echo "4. Value: (paste your FRED API key from .env)"
echo "5. Click 'Add secret'"
echo ""
echo "The daily GitHub Action will now run automatically at 07:00 UTC."
