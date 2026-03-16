#!/usr/bin/env bash

for FILE in config/issues_*.yaml
do
    echo "Processing $FILE"

    while IFS="|" read -r title milestone labels body
    do
        gh issue create \
            --title "$title" \
            --milestone "$milestone" \
            --label "$labels" \
            --body "$body"

    done < "$FILE"

done
