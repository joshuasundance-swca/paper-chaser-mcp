targetScope = 'resourceGroup'

@description('Azure region for all resources.')
param location string = resourceGroup().location

@description('Environment name used in resource names and tags, for example dev, staging, or prod.')
param environmentName string

@description('Short application name used in resource names.')
param appName string = 'scholar-search'

@description('Container image repository name inside ACR.')
param imageRepository string = 'scholar-search-mcp'

@description('Container image tag to deploy.')
param imageTag string = 'latest'

@description('Deployment mode. Use bootstrap to create shared infrastructure before the first workload rollout.')
@allowed([
  'bootstrap'
  'full'
])
param deployMode string = 'full'

@description('CPU allocated to the main Container App container.')
param containerCpu int = 1

@description('Memory allocated to the main Container App container.')
param containerMemory string = '2Gi'

@description('Minimum replica count. Use 1 or more for production to avoid cold starts.')
param minReplicas int = 1

@description('Maximum replica count.')
param maxReplicas int = 3

@description('Whether to enable the optional SerpApi provider path.')
param enableSerpApi bool = false

@description('Whether to enable additive smart workflows in the deployed server.')
param enableAgentic bool = false

@description('Provider bundle for the smart workflow layer.')
@allowed([
  'openai'
  'deterministic'
])
param agenticProvider string = 'openai'

@description('Planner model used for smart-search routing and query refinement.')
param plannerModel string = 'gpt-5.4-mini'

@description('Synthesis model used for grounded answers and theme labelling.')
param synthesisModel string = 'gpt-5.4'

@description('Embedding model used for smart ranking and saved-result-set retrieval.')
param embeddingModel string = 'text-embedding-3-large'

@description('Disable all embedding generation and embedding-based similarity paths.')
param disableEmbeddings bool = true

@description('Timeout in seconds for OpenAI-backed smart-layer requests.')
@minValue(1)
param agenticOpenAiTimeoutSeconds int = 30

@description('Workspace index backend for saved smart-search result sets.')
@allowed([
  'memory'
  'faiss'
])
param agenticIndexBackend string = 'memory'

@description('TTL in seconds for cached searchSessionId workspaces.')
@minValue(60)
param sessionTtlSeconds int = 1800

@description('Emit local trace events for smart workflows.')
param enableAgenticTraceLog bool = false

@description('Whether to enable the Semantic Scholar search provider.')
param enableSemanticScholar bool = true

@description('Whether to enable the arXiv search provider.')
param enableArxiv bool = true

@description('Whether to enable the CORE search provider. Off by default; requires a core-api-key Key Vault secret when enabled.')
param enableCore bool = false

@description('Whether to enable the OpenAlex search provider.')
param enableOpenAlex bool = true

@description('Whether to enable the Crossref provider for paper metadata and DOI lookup.')
param enableCrossref bool = true

@description('Whether to enable the Unpaywall provider for open-access full-text lookup.')
param enableUnpaywall bool = true

@description('Whether to enable the ECOS provider for species and environmental document retrieval.')
param enableEcos bool = true

@description('Whether to enable the Federal Register provider for regulatory document search.')
param enableFederalRegister bool = true

@description('Whether to enable the GovInfo CFR provider for Code of Federal Regulations retrieval. Requires a govinfo-api-key Key Vault secret for authoritative access; falls back to degraded HTML recovery without it.')
param enableGovinfoCfr bool = true

@description('Polite pool email sent as User-Agent to Crossref APIs. Optional; leave empty to use the application default.')
param crossrefMailto string = ''

@description('Email address sent as User-Agent to Unpaywall APIs. Optional; leave empty to use the application default.')
param unpayWallEmail string = ''

@description('Override the ECOS base URL. Leave empty to use the application default (https://ecos.fws.gov).')
param ecosBaseUrl string = ''

@description('Whether to verify TLS certificates when calling ECOS. Set to false only in air-gapped or private environments with custom PKI.')
param ecosVerifyTls bool = true

@description('Azure API Management v2 SKU used for private ingress plus outbound virtual network integration.')
@allowed([
  'StandardV2'
  'PremiumV2'
])
param apiManagementSku string = 'StandardV2'

@description('Relative API path exposed by API Management for the MCP service.')
param apiManagementApiPath string = 'scholar-search'

@description('Publisher display name shown in API Management.')
param apiManagementPublisherName string = 'Scholar Search MCP'

@description('Publisher email shown in API Management. Use a non-secret operational inbox.')
param apiManagementPublisherEmail string = 'owner@example.invalid'

@description('Safe tags applied to all resources. Do not place secrets or internal identifiers here.')
param tags object = {
  application: appName
  environment: environmentName
  managedBy: 'github-actions'
}

var suffix = toLower(replace('${appName}-${environmentName}', '_', '-'))
var deployFull = deployMode == 'full'
var workspaceName = 'law-${suffix}'
var appInsightsName = 'appi-${suffix}'
var acrName = take(replace('acr${suffix}', '-', ''), 50)
var keyVaultName = take(replace('kv-${suffix}', '_', '-'), 24)
var identityName = 'id-${suffix}'
var managedEnvironmentName = 'cae-${suffix}'
var containerAppName = 'aca-${suffix}'
var apiManagementName = take(replace('apim-${suffix}', '_', '-'), 50)
var backendAuthSecretName = 'mcp-backend-auth-token'
var apiManagementGatewayOrigin = 'https://${apiManagementName}.azure-api.net'

module network './modules/network.bicep' = {
  name: 'network'
  params: {
    environmentName: environmentName
    location: location
    name: suffix
    tags: tags
  }
}

module logAnalytics './modules/loganalytics.bicep' = {
  name: 'logAnalytics'
  params: {
    location: location
    name: workspaceName
    tags: tags
  }
}

module appInsights './modules/appinsights.bicep' = {
  name: 'applicationInsights'
  params: {
    location: location
    name: appInsightsName
    workspaceResourceId: logAnalytics.outputs.resourceId
    tags: tags
  }
}

module acr './modules/acr.bicep' = {
  name: 'acr'
  params: {
    location: location
    name: acrName
    privateDnsZoneId: network.outputs.acrPrivateDnsZoneId
    privateEndpointSubnetId: network.outputs.privateEndpointSubnetId
    tags: tags
  }
}

module identity './modules/managedIdentity.bicep' = {
  name: 'managedIdentity'
  params: {
    location: location
    name: identityName
    tags: tags
  }
}

module keyVault './modules/keyvault.bicep' = {
  name: 'keyVault'
  params: {
    location: location
    logAnalyticsWorkspaceResourceId: logAnalytics.outputs.resourceId
    name: keyVaultName
    privateDnsZoneId: network.outputs.keyVaultPrivateDnsZoneId
    privateEndpointSubnetId: network.outputs.privateEndpointSubnetId
    tenantId: subscription().tenantId
    tags: tags
  }
}

module roleAssignments './modules/roleAssignments.bicep' = {
  name: 'roleAssignments'
  params: {
    acrResourceId: acr.outputs.resourceId
    keyVaultResourceId: keyVault.outputs.resourceId
    principalId: identity.outputs.principalId
  }
}

module environment './modules/containerAppsEnv.bicep' = {
  name: 'containerAppsEnvironment'
  params: {
    appInsightsConnectionString: appInsights.outputs.connectionString
    infrastructureSubnetId: network.outputs.containerAppsInfrastructureSubnetId
    location: location
    logAnalyticsWorkspaceName: logAnalytics.outputs.name
    name: managedEnvironmentName
    privateDnsZoneId: network.outputs.containerAppsPrivateDnsZoneId
    privateEndpointSubnetId: network.outputs.privateEndpointSubnetId
    tags: tags
  }
}

module containerApp './modules/containerApp.bicep' = if (deployFull) {
  name: 'containerApp'
  dependsOn: [
    roleAssignments
  ]
  params: {
    appInsightsConnectionString: appInsights.outputs.connectionString
    agenticIndexBackend: agenticIndexBackend
    agenticProvider: agenticProvider
    allowedOrigins: apiManagementGatewayOrigin
    backendAuthSecretName: backendAuthSecretName
    backendAuthSecretUri: '${keyVault.outputs.vaultUri}secrets/${backendAuthSecretName}'
    containerAppName: containerAppName
    containerCpu: containerCpu
    containerMemory: containerMemory
    embeddingModel: embeddingModel
    disableEmbeddings: disableEmbeddings
    enableAgentic: enableAgentic
    enableAgenticTraceLog: enableAgenticTraceLog
    agenticOpenAiTimeoutSeconds: agenticOpenAiTimeoutSeconds
    environmentName: environmentName
    environmentResourceId: environment.outputs.resourceId
    image: '${acr.outputs.loginServer}/${imageRepository}:${imageTag}'
    keyVaultCoreApiKeySecretUri: '${keyVault.outputs.vaultUri}secrets/core-api-key'
    keyVaultOpenAiApiKeySecretUri: '${keyVault.outputs.vaultUri}secrets/openai-api-key'
    keyVaultOpenAlexApiKeySecretUri: '${keyVault.outputs.vaultUri}secrets/openalex-api-key'
    keyVaultOpenAlexMailtoSecretUri: '${keyVault.outputs.vaultUri}secrets/openalex-mailto'
    keyVaultSemanticScholarApiKeySecretUri: '${keyVault.outputs.vaultUri}secrets/semantic-scholar-api-key'
    keyVaultSerpApiKeySecretUri: '${keyVault.outputs.vaultUri}secrets/serpapi-api-key'
    keyVaultGovInfoApiKeySecretUri: '${keyVault.outputs.vaultUri}secrets/govinfo-api-key'
    location: location
    managedIdentityResourceId: identity.outputs.resourceId
    maxReplicas: maxReplicas
    minReplicas: minReplicas
    plannerModel: plannerModel
    registryServer: acr.outputs.loginServer
    sessionTtlSeconds: sessionTtlSeconds
    synthesisModel: synthesisModel
    tags: tags
    enableSerpApi: enableSerpApi
    enableSemanticScholar: enableSemanticScholar
    enableArxiv: enableArxiv
    enableCore: enableCore
    enableOpenAlex: enableOpenAlex
    enableCrossref: enableCrossref
    enableUnpaywall: enableUnpaywall
    enableEcos: enableEcos
    enableFederalRegister: enableFederalRegister
    enableGovinfoCfr: enableGovinfoCfr
    crossrefMailto: crossrefMailto
    unpayWallEmail: unpayWallEmail
    ecosBaseUrl: ecosBaseUrl
    ecosVerifyTls: ecosVerifyTls
  }
}

module apiManagement './modules/apim.bicep' = if (deployFull) {
  name: 'apiManagement'
  dependsOn: [
    roleAssignments
  ]
  params: {
    apiPath: apiManagementApiPath
    backendAuthNamedValueName: 'mcp-backend-auth-token'
    backendAuthSecretUri: '${keyVault.outputs.vaultUri}secrets/${backendAuthSecretName}'
    backendUrl: 'https://${containerApp.outputs.fqdn}${containerApp.outputs.mcpPath}'
    integrationSubnetId: network.outputs.apiManagementSubnetId
    keyVaultResourceId: keyVault.outputs.resourceId
    location: location
    name: apiManagementName
    privateDnsZoneId: network.outputs.apiManagementPrivateDnsZoneId
    privateEndpointSubnetId: network.outputs.privateEndpointSubnetId
    publisherEmail: apiManagementPublisherEmail
    publisherName: apiManagementPublisherName
    skuName: apiManagementSku
    tags: tags
  }
}

output apiManagementGatewayUrl string = deployFull ? apiManagement!.outputs.gatewayUrl : ''
output containerAppFqdn string = deployFull ? containerApp!.outputs.fqdn : ''
output containerAppHealthUrl string = deployFull ? 'https://${containerApp!.outputs.fqdn}/healthz' : ''
output activeDeployMode string = deployMode
output keyVaultName string = keyVault.outputs.name
output managedIdentityResourceId string = identity.outputs.resourceId
