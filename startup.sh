#!/usr/bin/env bash
cd /usr/local/classifier_HR/api
pipenv run python3 hr_pred_api.py
uwsgi --ini /var/www/app/uwsgi.ini
