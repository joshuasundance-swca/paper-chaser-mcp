using './main.bicep'

param environmentName = 'dev'
param imageTag = 'latest'
param minReplicas = 0
param maxReplicas = 2
param apiManagementSku = 'StandardV2'
