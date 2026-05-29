#!/usr/bin/env bash
# RunPod container entrypoint.
#  * Installs the pod's injected public key so RunPod's ssh.runpod.io works.
#  * Starts sshd.
#  * Keeps the container alive (web terminal / exec attach into this process).
set -e

mkdir -p /run/sshd ~/.ssh
chmod 700 ~/.ssh
if [ -n "${PUBLIC_KEY:-}" ]; then
    echo "${PUBLIC_KEY}" >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
fi

if command -v sshd >/dev/null 2>&1; then
    /usr/sbin/sshd
fi

echo "Pod ready. Project at /opt/algae_detection, weights at /opt/models."
exec sleep infinity
