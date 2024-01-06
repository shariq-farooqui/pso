FROM python:3.10-slim-bullseye as local

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["streamlit", "run", "Home.py", \
     "--server.port", "8501", \
     "--server.headless", "true", \
     "--server.enableCORS", "true", \
     "--browser.gatherUsageStats", "false", \
     "--theme.base", "light", \
     "--server.runOnSave", "true", \
     "--theme.primaryColor", "#008BC8"]
