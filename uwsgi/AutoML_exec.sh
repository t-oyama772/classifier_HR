#!/usr/bin/env bash
cd classifier_HR
pipenv install
cd bin
pipenv run python3 classifier_HR.py
