#!/usr/bin/env bash

FILE=config/labels.yaml

while IFS="|" read -r name color description
do
    if gh label list | grep -q "$name"; then
        echo "Label exists: $name"
    else
        gh label create "$name" \
            --color "$color" \
            --description "$description"
    fi
done < "$FILE"
