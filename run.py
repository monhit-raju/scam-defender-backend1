from app import create_app
import os

# Create app for Gunicorn
app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render default
    app.run(host="0.0.0.0", port=port)
