@echo off
echo Starting Clout Detective backend...
echo.
echo 1. Install dependencies (if needed):
echo    pip install -r backend\requirements.txt
echo.
echo 2. Starting Flask server on http://localhost:5000
echo    (Training takes ~5-10 seconds on first run)
echo.
cd backend
python app.py
