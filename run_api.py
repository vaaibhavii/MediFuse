"""
Click the "Run" (play) button on THIS file in VS Code to start the backend
server — no terminal typing needed.

Once it's running, open this in your browser:
    http://127.0.0.1:8000/docs

That page lets you test /predict and /urine/analyze by clicking buttons and
filling in a form — you never need curl or Postman.

To stop the server: click the trash-can / stop icon on the terminal panel
that opens at the bottom of VS Code.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
