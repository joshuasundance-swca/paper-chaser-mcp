using './main.bicep'

param environmentName = 'prod'
param imageTag = 'latest'
param minReplicas = 1
param maxReplicas = 5
param apiManagementSku = 'StandardV2'
