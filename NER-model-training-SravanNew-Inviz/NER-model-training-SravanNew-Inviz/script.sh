touch temp.json
echo $gcp_credential > temp.json
sudo snap install google-cloud-cli --classic
export SERVICE_ACCOUNT='vertex-ai-service-account@search-sandbox-363805.iam.gserviceaccount.com'
export LOCATION='asia-south1'
export JOB_NAME='ner-model-test'
export MACHINE_TYPE='n1-highmem-16'
export REPLICA_COUNT='1'
# export CUSTOM_CONTAINER_IMAGE_URI='gcr.io/search-sandbox-363805/ner-model-train:latest'
export PROJECT='search-sandbox-363805'
# export APP_ENV='b9dc3c08-b967-4737-8280-86e065413b68'
# export MODEL_ENV='ner20c08-b967-4737-8280-86e065413b68'
gcloud auth activate-service-account $SERVICE_ACCOUNT --key-file=temp.json --project=$PROJECT

gcloud ai custom-jobs create \
  --region=$LOCATION \
  --display-name=$JOB_NAME \
  --config=config.yaml \
  --service-account=$SERVICE_ACCOUNT
  #--worker-pool-spec=machine-type=$MACHINE_TYPE,replica-count=$REPLICA_COUNT,container-image-uri=$CUSTOM_CONTAINER_IMAGE_URI \
 # --args={'APP_ID':'b9dc3c08-b967-4737-8280-86e065413b68','MODEL_ID':'ner20c08-b967-4737-8280-86e065413b68'}



