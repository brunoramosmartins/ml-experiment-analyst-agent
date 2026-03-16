#!/usr/bin/env bash

FILE=config/milestones.yaml

while IFS="|" read -r title description
do
    if gh api repos/:owner/:repo/milestones | grep -q "$title"; then
        echo "Milestone already exists: $title"
    else
        gh api repos/:owner/:repo/milestones \
            -f title="$title" \
            -f description="$description"
    fi
done < "$FILE"
