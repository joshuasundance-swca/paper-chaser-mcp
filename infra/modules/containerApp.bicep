param appInsightsConnectionString string
param agenticIndexBackend string
param agenticOpenAiTimeoutSeconds int
param agenticProvider string
param allowedOrigins string
param backendAuthSecretName string
param backendAuthSecretUri string
param containerAppName string
param containerCpu int
param containerMemory string
param disableEmbeddings bool
param embeddingModel string
param enableAgentic bool
param enableAgenticTraceLog bool
param enableSerpApi bool
param environmentName string
param environmentResourceId string
param image string
param keyVaultCoreApiKeySecretUri string
param keyVaultOpenAiApiKeySecretUri string
param keyVaultOpenAlexApiKeySecretUri string
param keyVaultOpenAlexMailtoSecretUri string
param keyVaultSemanticScholarApiKeySecretUri string
param keyVaultSerpApiKeySecretUri string
param location string
param managedIdentityResourceId string
param maxReplicas int
param minReplicas int
param plannerModel string
param registryServer string
param sessionTtlSeconds int
param synthesisModel string
param tags object = {}

var secretDefinitions = concat([
  {
    identity: managedIdentityResourceId
    keyVaultUrl: backendAuthSecretUri
    name: backendAuthSecretName
  }
  {
    identity: managedIdentityResourceId
    keyVaultUrl: keyVaultCoreApiKeySecretUri
    name: 'core-api-key'
  }
  {
    identity: managedIdentityResourceId
    keyVaultUrl: keyVaultSemanticScholarApiKeySecretUri
    name: 'semantic-scholar-api-key'
  }
  {
    identity: managedIdentityResourceId
    keyVaultUrl: keyVaultOpenAlexApiKeySecretUri
    name: 'openalex-api-key'
  }
  {
    identity: managedIdentityResourceId
    keyVaultUrl: keyVaultOpenAlexMailtoSecretUri
    name: 'openalex-mailto'
  }
], enableSerpApi ? [
  {
    identity: managedIdentityResourceId
    keyVaultUrl: keyVaultSerpApiKeySecretUri
    name: 'serpapi-api-key'
  }
] : [], enableAgentic && agenticProvider == 'openai' ? [
  {
    identity: managedIdentityResourceId
    keyVaultUrl: keyVaultOpenAiApiKeySecretUri
    name: 'openai-api-key'
  }
] : [])

var containerEnv = concat([
  {
    name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
    value: appInsightsConnectionString
  }
  {
    name: 'CORE_API_KEY'
    secretRef: 'core-api-key'
  }
  {
    name: 'OPENALEX_API_KEY'
    secretRef: 'openalex-api-key'
  }
  {
    name: 'OPENALEX_MAILTO'
    secretRef: 'openalex-mailto'
  }
  {
    name: 'PORT'
    value: '8080'
  }
  {
    name: 'SCHOLAR_SEARCH_ALLOWED_ORIGINS'
    value: allowedOrigins
  }
  {
    name: 'SCHOLAR_SEARCH_ENABLE_ARXIV'
    value: 'true'
  }
  {
    name: 'SCHOLAR_SEARCH_ENABLE_CORE'
    value: 'true'
  }
  {
    name: 'SCHOLAR_SEARCH_ENABLE_OPENALEX'
    value: 'true'
  }
  {
    name: 'SCHOLAR_SEARCH_ENABLE_SEMANTIC_SCHOLAR'
    value: 'true'
  }
  {
    name: 'SCHOLAR_SEARCH_ENABLE_SERPAPI'
    value: string(enableSerpApi)
  }
  {
    name: 'SCHOLAR_SEARCH_ENABLE_AGENTIC'
    value: string(enableAgentic)
  }
  {
    name: 'SCHOLAR_SEARCH_AGENTIC_PROVIDER'
    value: agenticProvider
  }
  {
    name: 'SCHOLAR_SEARCH_PLANNER_MODEL'
    value: plannerModel
  }
  {
    name: 'SCHOLAR_SEARCH_SYNTHESIS_MODEL'
    value: synthesisModel
  }
  {
    name: 'SCHOLAR_SEARCH_EMBEDDING_MODEL'
    value: embeddingModel
  }
  {
    name: 'SCHOLAR_SEARCH_DISABLE_EMBEDDINGS'
    value: string(disableEmbeddings)
  }
  {
    name: 'SCHOLAR_SEARCH_AGENTIC_OPENAI_TIMEOUT_SECONDS'
    value: string(agenticOpenAiTimeoutSeconds)
  }
  {
    name: 'SCHOLAR_SEARCH_AGENTIC_INDEX_BACKEND'
    value: agenticIndexBackend
  }
  {
    name: 'SCHOLAR_SEARCH_SESSION_TTL_SECONDS'
    value: string(sessionTtlSeconds)
  }
  {
    name: 'SCHOLAR_SEARCH_ENABLE_AGENTIC_TRACE_LOG'
    value: string(enableAgenticTraceLog)
  }
  {
    name: 'SCHOLAR_SEARCH_HTTP_AUTH_HEADER'
    value: 'x-backend-auth'
  }
  {
    name: 'SCHOLAR_SEARCH_HTTP_AUTH_TOKEN'
    secretRef: backendAuthSecretName
  }
  {
    name: 'SCHOLAR_SEARCH_HTTP_HOST'
    value: '0.0.0.0'
  }
  {
    name: 'SCHOLAR_SEARCH_HTTP_PATH'
    value: '/mcp'
  }
  {
    name: 'SCHOLAR_SEARCH_HTTP_PORT'
    value: '8080'
  }
  {
    name: 'SCHOLAR_SEARCH_TRANSPORT'
    value: 'streamable-http'
  }
  {
    name: 'SEMANTIC_SCHOLAR_API_KEY'
    secretRef: 'semantic-scholar-api-key'
  }
], enableSerpApi ? [
  {
    name: 'SERPAPI_API_KEY'
    secretRef: 'serpapi-api-key'
  }
] : [], enableAgentic && agenticProvider == 'openai' ? [
  {
    name: 'OPENAI_API_KEY'
    secretRef: 'openai-api-key'
  }
] : [])

resource app 'Microsoft.App/containerApps@2025-07-01' = {
  name: containerAppName
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${managedIdentityResourceId}': {}
    }
  }
  properties: {
    environmentId: environmentResourceId
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        allowInsecure: false
        external: true
        targetPort: 8080
        transport: 'http'
      }
      registries: [
        {
          identity: managedIdentityResourceId
          server: registryServer
        }
      ]
      secrets: secretDefinitions
    }
    template: {
      containers: [
        {
          args: [
            'deployment-http'
          ]
          env: containerEnv
          image: image
          name: 'scholar-search'
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path: '/healthz'
                port: 8080
              }
              initialDelaySeconds: 10
              periodSeconds: 15
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/healthz'
                port: 8080
              }
              initialDelaySeconds: 5
              periodSeconds: 10
            }
          ]
          resources: {
            cpu: containerCpu
            memory: containerMemory
          }
        }
      ]
      scale: {
        maxReplicas: maxReplicas
        minReplicas: minReplicas
      }
    }
  }
  tags: union(tags, {
    deploymentMode: 'private-api-behind-private-apim'
    publicRepoSafe: 'true'
    stage: environmentName
  })
}

output fqdn string = app.properties.configuration.ingress.fqdn
output mcpPath string = '/mcp'
output resourceId string = app.id
