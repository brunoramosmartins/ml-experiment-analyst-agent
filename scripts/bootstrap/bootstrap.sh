#!/usr/bin/env bash

set -e

echo "Starting repository bootstrap..."

if ! command -v gh &> /dev/null
then
    echo "GitHub CLI not installed."
    exit
fi

gh auth status > /dev/null || gh auth login

echo "Creating milestones..."
bash scripts/bootstrap/create_milestones.sh

echo "Creating labels..."
bash scripts/bootstrap/create_labels.sh

echo "Creating issue templates..."
bash scripts/bootstrap/create_templates.sh

echo "Creating issues from roadmap..."
bash scripts/bootstrap/create_issues.sh

echo "Bootstrap finished."
