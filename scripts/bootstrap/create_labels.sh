#!/usr/bin/env bash

FILE=config/labels.yaml

while IFS="|" read -r name color description
do
    if gh label list --json name --jq '.[].name' | grep -qx "$name"; then
        echo "Label exists: $name"
    else
        gh label create "$name" \
            --color "$color" \
            --description "$description"
    fi
done < "$FILE"
