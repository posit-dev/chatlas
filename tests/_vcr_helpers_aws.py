"""AWS-specific VCR scrubbing functions for Bedrock tests."""

import json
import re

from .conftest import _filter_response_headers


def _scrub_aws_credentials(response):
    """Scrub AWS credentials from response body before recording."""
    body = response.get("body", {})
    if not body:
        return response

    body_string = body.get("string", "")
    if not body_string:
        return response

    # Handle both string and bytes
    if isinstance(body_string, bytes):
        try:
            body_string = body_string.decode("utf-8")
            was_bytes = True
        except UnicodeDecodeError:
            return response
    else:
        was_bytes = False

    # Try to parse as JSON and scrub credentials
    try:
        data = json.loads(body_string)
        if "roleCredentials" in data:
            creds = data["roleCredentials"]
            if "accessKeyId" in creds:
                creds["accessKeyId"] = "SCRUBBED_ACCESS_KEY_ID"
            if "secretAccessKey" in creds:
                creds["secretAccessKey"] = "SCRUBBED_SECRET_ACCESS_KEY"
            if "sessionToken" in creds:
                creds["sessionToken"] = "SCRUBBED_SESSION_TOKEN"
            body_string = json.dumps(data)
    except (json.JSONDecodeError, TypeError):
        # Not JSON, try regex patterns for AWS credentials
        # Access key pattern: AKIA... or ASIA... (20 chars)
        body_string = re.sub(
            r"(A[SK]IA[A-Z0-9]{16})",
            "SCRUBBED_ACCESS_KEY_ID",
            body_string,
        )
        # Secret key pattern (40 chars of base64-ish characters after common prefixes)
        body_string = re.sub(
            r'("secretAccessKey"\s*:\s*")[^"]+(")',
            r"\1SCRUBBED_SECRET_ACCESS_KEY\2",
            body_string,
        )
        # Session token pattern
        body_string = re.sub(
            r'("sessionToken"\s*:\s*")[^"]+(")',
            r"\1SCRUBBED_SESSION_TOKEN\2",
            body_string,
        )

    if was_bytes:
        body["string"] = body_string.encode("utf-8")
    else:
        body["string"] = body_string

    return response


def _scrub_aws_request(request):
    """Scrub AWS account ID from request URLs."""
    # Replace account_id parameter in SSO URLs
    if "account_id=" in request.uri:
        request.uri = re.sub(
            r"account_id=\d+",
            "account_id=SCRUBBED_ACCOUNT_ID",
            request.uri,
        )
    return request


def _filter_aws_response(response):
    """Combined filter for response headers and AWS credentials."""
    response = _filter_response_headers(response)
    response = _scrub_aws_credentials(response)
    return response
