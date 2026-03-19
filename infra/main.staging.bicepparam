using './main.bicep'

param environmentName = 'staging'
param imageTag = 'latest'
param minReplicas = 1
param maxReplicas = 3
param apiManagementSku = 'StandardV2'
