import re
from datetime import date


def validate_email(email: str) -> bool:
    pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email)) and len(email) <= 255


def validate_password_strength(password: str) -> tuple[bool, str]:
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter."
    if not re.search(r"\d", password):
        return False, "Password must contain at least one digit."
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least one special character."
    return True, ""


def validate_vehicle_year(year) -> tuple[bool, str]:
    try:
        year = int(year)
    except (TypeError, ValueError):
        return False, "Vehicle year must be an integer."
    current_year = date.today().year
    if year < 1900 or year > current_year + 1:
        return False, f"Vehicle year must be between 1900 and {current_year + 1}."
    return True, ""


def validate_claim_amount(amount) -> tuple[bool, str]:
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return False, "Claim amount must be a number."
    if amount <= 0:
        return False, "Claim amount must be greater than 0."
    if amount > 100_000_000:
        return False, "Claim amount exceeds maximum allowed value."
    return True, ""


def validate_incident_date(date_str: str) -> tuple[bool, str]:
    try:
        parsed = date.fromisoformat(date_str)
    except (ValueError, TypeError):
        return False, "Incident date must be in YYYY-MM-DD format."
    if parsed > date.today():
        return False, "Incident date cannot be in the future."
    return True, ""


def validate_vehicle_number(number: str) -> tuple[bool, str]:
    if not number or not isinstance(number, str):
        return False, "Vehicle number is required."
    number = number.strip()
    if len(number) < 2 or len(number) > 20:
        return False, "Vehicle number must be between 2 and 20 characters."
    if not re.match(r"^[A-Za-z0-9\-\s]+$", number):
        return False, "Vehicle number contains invalid characters."
    return True, ""
