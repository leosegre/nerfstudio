#!/bin/bash
set -e

# Perform a git pull in your repository
git fetch

# Run your main application command
exec "$@"