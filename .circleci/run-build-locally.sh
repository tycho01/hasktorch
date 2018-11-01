#!/usr/bin/env bash
COMMIT_HASH="$(git rev-parse HEAD)"
HASKTORCH_FORK="$(whoami)"
BRANCH_NAME="dev"

curl --user ${CIRCLE_TOKEN}: \
    --request POST \
    --form revision=${COMMIT_HASH} \
    --form config=@config.yml \
    --form notify=false \
        https://circleci.com/api/v1.1/project/github/${HASKTORCH_FORK}/hasktorch/tree/${BRANCH_NAME}
