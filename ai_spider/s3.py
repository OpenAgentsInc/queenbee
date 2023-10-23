import os

import boto3

g_s3 = None


def s3():
    global g_s3
    if not g_s3:
        g_s3 = boto3.client('s3',
                            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"))

    return g_s3
