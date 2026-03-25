using './main.bicep'

param environmentName = 'staging'
param imageTag = 'latest'
param minReplicas = 1
param maxReplicas = 3
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
param enableFederalRegister = true
param enableGovinfoCfr = true
