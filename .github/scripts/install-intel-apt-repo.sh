#!/bin/bash
# Register the Intel oneAPI APT repository so its packages can be installed.
# See: https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2024-0/apt.html

set -euo pipefail

success=false
for i in 1 2 3 4 5; do
  if wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null; then
    success=true
    break
  fi
  if [ "$i" -lt 5 ]; then
    echo "Attempt $i failed, retrying in 10s..."
    sleep 10
  else
    echo "Attempt $i failed; no more retries left."
  fi
done
if [ "$success" = false ]; then
  echo "Failed to download Intel GPG key after 5 attempts"
  exit 1
fi
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt-get update -y
