#!/bin/bash
set -e

# Perform a git pull in your repository
cd ..
rm -rf nerfstudio
git clone -b registration https://github.com/leosegre/nerfstudio.git
cd nerfstudio

# Run your main application command
exec "$@"