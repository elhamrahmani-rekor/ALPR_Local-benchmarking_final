from datetime import datetime
import json
import logging
import os
import pickle
import re
import requests

logger = logging.getLogger(__name__)

SLACK_WEBHOOK_URL = 'https://hooks.slack.com/services/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'

def message_slack(msg, subject=''):
    """Post to the #ml-pipeline channel via Tag Server app

    :param str msg: Text to send to channel
    :param str subject: Optional header line (if not empty, message will be
        preceded by a line break).
    :return: None
    """

    endpoint = SLACK_WEBHOOK_URL
    if subject == '':
        payload = {'text': msg}
    else:
        payload = {'text': '{}\n{}'.format(subject, msg)}
    r = requests.post(endpoint, data=json.dumps(payload), headers={'Content-Type': 'application/json'})
    if not r.ok:
        logger.error('Bad status code {} for Slack message'.format(r.status_code))


