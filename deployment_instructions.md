# Deployment Instructions for MCP Analytical System

This document provides instructions for deploying the MCP Analytical System in various environments.

## Prerequisites

Before deploying the system, ensure you have the following:

- Python 3.8 or higher
- Redis server (for context storage)
- API keys for external services:
  - Perplexity API key for Sonar
  - Groq API key for Llama 4 Scout
  - Brave Search API key
  - Academic Search API key
  - FRED API key (for economic data)
  - World Bank API key (for economic data)
  - IMF API key (for economic data)
  - GDELT API key (for geopolitical data)
  - ACLED API key (for geopolitical data)

## Local Deployment

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file with your API keys:

```
PERPLEXITY_API_KEY=your_perplexity_api_key
GROQ_API_KEY=your_groq_api_key
BRAVE_API_KEY=your_brave_api_key
ACADEMIC_API_KEY=your_academic_api_key
FRED_API_KEY=your_fred_api_key
WORLD_BANK_API_KEY=your_world_bank_api_key
IMF_API_KEY=your_imf_api_key
GDELT_API_KEY=your_gdelt_api_key
ACLED_API_KEY=your_acled_api_key
REDIS_URL=redis://localhost:6379/0
```

### 3. Start Redis Server

```bash
redis-server
```

### 4. Run the Streamlit Application

```bash
streamlit run src/app.py
```

The application will be available at http://localhost:8501.

## Docker Deployment

### 1. Build Docker Image

```bash
docker build -t mcp-analytical-system .
```

### 2. Run Docker Container

```bash
docker run -p 8501:8501 --env-file .env mcp-analytical-system
```

The application will be available at http://localhost:8501.

## Cloud Deployment

### Heroku Deployment

1. Install Heroku CLI:
   ```bash
   npm install -g heroku
   ```

2. Login to Heroku:
   ```bash
   heroku login
   ```

3. Create a new Heroku app:
   ```bash
   heroku create mcp-analytical-system
   ```

4. Add Redis add-on:
   ```bash
   heroku addons:create heroku-redis:hobby-dev
   ```

5. Set environment variables:
   ```bash
   heroku config:set PERPLEXITY_API_KEY=your_perplexity_api_key
   heroku config:set GROQ_API_KEY=your_groq_api_key
   # Set other API keys...
   ```

6. Deploy to Heroku:
   ```bash
   git push heroku main
   ```

7. Open the application:
   ```bash
   heroku open
   ```

### AWS Elastic Beanstalk Deployment

1. Install AWS CLI and EB CLI:
   ```bash
   pip install awscli awsebcli
   ```

2. Configure AWS credentials:
   ```bash
   aws configure
   ```

3. Initialize EB application:
   ```bash
   eb init -p python-3.8 mcp-analytical-system
   ```

4. Create environment:
   ```bash
   eb create mcp-analytical-system-env
   ```

5. Set environment variables:
   ```bash
   eb setenv PERPLEXITY_API_KEY=your_perplexity_api_key GROQ_API_KEY=your_groq_api_key
   # Set other API keys...
   ```

6. Deploy the application:
   ```bash
   eb deploy
   ```

7. Open the application:
   ```bash
   eb open
   ```

### Google Cloud Run Deployment

1. Install Google Cloud SDK:
   ```bash
   # Follow instructions at https://cloud.google.com/sdk/docs/install
   ```

2. Login to Google Cloud:
   ```bash
   gcloud auth login
   ```

3. Set project:
   ```bash
   gcloud config set project your-project-id
   ```

4. Build and push Docker image:
   ```bash
   gcloud builds submit --tag gcr.io/your-project-id/mcp-analytical-system
   ```

5. Deploy to Cloud Run:
   ```bash
   gcloud run deploy mcp-analytical-system \
     --image gcr.io/your-project-id/mcp-analytical-system \
     --platform managed \
     --set-env-vars PERPLEXITY_API_KEY=your_perplexity_api_key,GROQ_API_KEY=your_groq_api_key
   # Set other API keys...
   ```

6. Get the service URL:
   ```bash
   gcloud run services describe mcp-analytical-system --platform managed --format 'value(status.url)'
   ```

## Streamlit Cloud Deployment

1. Create a GitHub repository with your code.

2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign in.

3. Click "New app" and select your repository.

4. Set the main file path to `src/app.py`.

5. Add your API keys as secrets in the Streamlit Cloud dashboard.

6. Deploy the app.

## Environment Variables

The following environment variables are used by the system:

| Variable | Description |
|----------|-------------|
| PERPLEXITY_API_KEY | API key for Perplexity Sonar |
| GROQ_API_KEY | API key for Groq (Llama 4 Scout) |
| BRAVE_API_KEY | API key for Brave Search |
| ACADEMIC_API_KEY | API key for Academic Search |
| FRED_API_KEY | API key for FRED economic data |
| WORLD_BANK_API_KEY | API key for World Bank data |
| IMF_API_KEY | API key for IMF data |
| GDELT_API_KEY | API key for GDELT geopolitical data |
| ACLED_API_KEY | API key for ACLED conflict data |
| REDIS_URL | URL for Redis server |

## Testing the Deployment

After deployment, you can test the system by:

1. Opening the web interface
2. Entering an analytical question
3. Observing the system's analysis process
4. Reviewing the final results

## Troubleshooting

### Common Issues

- **API Connection Errors**: Verify API keys are correctly set in environment variables.
- **Redis Connection Errors**: Ensure Redis server is running and accessible.
- **Memory Issues**: For complex analyses, ensure sufficient memory is allocated.
- **Timeout Errors**: For cloud deployments, adjust timeout settings if analyses take too long.

### Logs

- **Local**: Check console output and `logs/` directory
- **Docker**: `docker logs container_id`
- **Heroku**: `heroku logs --tail`
- **AWS**: `eb logs`
- **GCP**: `gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=mcp-analytical-system"`
- **Streamlit Cloud**: Check logs in the Streamlit Cloud dashboard

## Scaling Considerations

For production deployments with high traffic:

1. Use a managed Redis service with appropriate capacity
2. Consider horizontal scaling for the web application
3. Implement caching for common analyses
4. Monitor API usage to avoid hitting rate limits

## Security Considerations

1. Store API keys securely using environment variables or secret management services
2. Implement authentication for the web interface in production
3. Use HTTPS for all communications
4. Regularly update dependencies to address security vulnerabilities
5. Consider data privacy implications when storing analysis results
