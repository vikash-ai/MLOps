# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.9-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Set the working directory to /MLOps-main
WORKDIR /MLOps-main

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# Set the working directory to /app
WORKDIR /app

# Copy the contents of the local folder to the working directory in the container
COPY . /app

EXPOSE 8501

# To test the container, that it is still working
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# During debugging, this entry point will be overridden.
# For more information, please refer to https://aka.ms/vscode-docker-python-debug
ENTRYPOINT ["streamlit", "run", "FairLendingRiskAssessment.py", "--server.port=8501", "--server.address=0.0.0.0"]

