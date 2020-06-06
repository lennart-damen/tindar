# Before we can run our app, we need to create an entrypoint.
# This is where we'll instruct our app to run.

# We're importing the app variable from the app package that we've just created.
from app.app import app

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)