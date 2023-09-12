#!/bin/bash
set -e

# Perform a git pull in your repository
git reset --hard
git clean -fd
git pull

# Run your main application command
exec "$@"