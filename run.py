from app import create_app
import os

if __name__ == "__main__":
    app = create_app()
    # Allow access from any device on network
    app.run(host="0.0.0.0", port=5000, debug=True)
