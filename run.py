from flask_application import app
import os

if __name__ == "__main__":
    try:
        # If Flask is running on Docker,
        # use host and port specified in Docker image
        if int(os.environ["DOCKER"]) == 1:
            app.run(
                host=os.environ["HOST"],
                port=int(os.environ["PORT"]),
                debug=True,
                threaded=True
            )
    except KeyError:
        # default config
        app.run(host="0.0.0.0", port="8080",
                debug=True, threaded=True)
