# Use an official base image of Python 3.12.5
FROM python:3.12.5-slim
# Create the working directory
RUN mkdir /video-analysis-with-gpt-4o
# Set the working directory to /video-analysis-with-gpt-4o
WORKDIR /video-analysis-with-gpt-4o
# Copy the requirements files into the image
COPY requirements.txt /video-analysis-with-gpt-4o/requirements.txt
# Install the dependencies
RUN pip install --no-cache-dir -r /video-analysis-with-gpt-4o/requirements.txt
# Copy the rest of the application files
COPY . .
# Expose the port on which the application will run
EXPOSE 8501
# Define the default command to run the application
ENTRYPOINT ["streamlit", "run", "video-analysis-with-gpt-4o.py", "--server.port=8501", "--server.address=0.0.0.0"]