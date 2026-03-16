#!/usr/bin/env bash

echo "Creating GitHub templates..."

mkdir -p .github/ISSUE_TEMPLATE

cat <<EOF > .github/ISSUE_TEMPLATE/task.md
---
name: Task
about: Development task
---

## Context

Describe why this task exists.

## Tasks

- [ ] Task 1
- [ ] Task 2

## Definition of Done

- [ ] Criterion 1
- [ ] Criterion 2
EOF

cat <<EOF > .github/ISSUE_TEMPLATE/bug.md
---
name: Bug report
about: Report a bug
---

## Description

Describe the issue.

## Steps to reproduce

1.
2.

## Expected behavior

Explain what should happen.
EOF

cat <<EOF > .github/pull_request_template.md
## Context

Describe the change.

## Objective

What engineering problem does this PR solve?

## Scope

- [ ] Feature implemented
- [ ] Tests added
- [ ] Documentation updated

## Validation

- [ ] Tests pass
- [ ] CI passing
EOF

echo "Templates created."
