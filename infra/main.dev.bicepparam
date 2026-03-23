using './main.bicep'

param environmentName = 'dev'
param imageTag = 'latest'
param minReplicas = 0
param maxReplicas = 2
param apiManagementSku = 'StandardV2'

// Provider enable flags — explicit to document intent and keep defaults obvious.
// CORE is off by default (matches app default); enable and seed core-api-key in
// Key Vault only when you want the CORE fallback hop.
param enableCore = false
param enableSemanticScholar = true
param enableArxiv = true
param enableOpenAlex = true
param enableCrossref = true
param enableUnpaywall = true
param enableEcos = true
