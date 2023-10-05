import os

from dotenv import load_dotenv

load_dotenv()


def get_bill_to(request):
    req_user = request.headers.get("Authorization")
    bill_to_token = ""
    if req_user and " " in req_user:
        bill_to_token = req_user.split(" ")[1]
    return bill_to_token


BILLING_URL = os.environ["BILLING_URL"]
BILLING_TIMEOUT = 20
