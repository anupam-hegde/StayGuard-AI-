# Use Jenkins LTS
FROM jenkins/jenkins:lts

# Switch to root to install packages
USER root

# Install Docker CLI (NOT Docker Engine)
RUN apt-get update && \
    apt-get install -y docker.io && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Add jenkins user to docker group
RUN usermod -aG docker jenkins

# Switch back to jenkins user
USER jenkins
