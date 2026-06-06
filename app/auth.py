from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

USERS = {
    "admin": "admin123"
}

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    correct_password = USERS.get(credentials.username)

    if not correct_password or correct_password != credentials.password:
        raise HTTPException(status_code=401, detail="Unauthorized")

    return credentials.username