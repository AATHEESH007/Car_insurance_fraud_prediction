from flask import jsonify, g


def success_response(data=None, message=None, status_code=200):
    body = {"success": True}
    if data is not None:
        body["data"] = data
    if message:
        body["message"] = message
    resp = jsonify(body)
    resp.status_code = status_code
    _attach_request_id(resp)
    return resp


def error_response(code, message, status_code, details=None):
    body = {"success": False, "error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    resp = jsonify(body)
    resp.status_code = status_code
    _attach_request_id(resp)
    return resp


def _attach_request_id(response):
    request_id = getattr(g, "request_id", None)
    if request_id:
        response.headers["X-Request-ID"] = request_id
    return response
