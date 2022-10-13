# get project-ID2:
#gcloud config get-value project
PROJECT_ID='stock-363101'
### build docker and push into GCP
gcloud builds submit --tag gcr.io/${PROJECT_ID}/stockpredictor

### deploy image in GG RUN:
gcloud run deploy --image gcr.io/${PROJECT_ID}/stockpredictor --platform managed
