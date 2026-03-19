param apiPath string
param backendAuthNamedValueName string
param backendAuthSecretUri string
param backendUrl string
param integrationSubnetId string
param keyVaultResourceId string
param location string
param name string
param privateDnsZoneId string
param privateEndpointSubnetId string
param publisherEmail string
param publisherName string
param skuName string
param tags object = {}

var keyVaultName = last(split(keyVaultResourceId, '/'))
var keyVaultSecretsUserRoleDefinitionId = subscriptionResourceId(
  'Microsoft.Authorization/roleDefinitions',
  '4633458b-17de-408a-b874-0445c86b69e6'
)
var operationDefinitions = [
  {
    displayName: 'Delete root'
    method: 'DELETE'
    name: 'delete-root'
    urlTemplate: '/'
  }
  {
    displayName: 'Delete wildcard'
    method: 'DELETE'
    name: 'delete-wildcard'
    urlTemplate: '/*'
  }
  {
    displayName: 'Get root'
    method: 'GET'
    name: 'get-root'
    urlTemplate: '/'
  }
  {
    displayName: 'Get wildcard'
    method: 'GET'
    name: 'get-wildcard'
    urlTemplate: '/*'
  }
  {
    displayName: 'Post root'
    method: 'POST'
    name: 'post-root'
    urlTemplate: '/'
  }
  {
    displayName: 'Post wildcard'
    method: 'POST'
    name: 'post-wildcard'
    urlTemplate: '/*'
  }
]

resource service 'Microsoft.ApiManagement/service@2024-10-01-preview' = {
  name: name
  location: location
  sku: {
    capacity: 1
    name: skuName
  }
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    customProperties: {
      'Microsoft.WindowsAzure.ApiManagement.Gateway.Security.Backend.Protocols.Ssl30': 'false'
      'Microsoft.WindowsAzure.ApiManagement.Gateway.Security.Backend.Protocols.Tls10': 'false'
      'Microsoft.WindowsAzure.ApiManagement.Gateway.Security.Backend.Protocols.Tls11': 'false'
      'Microsoft.WindowsAzure.ApiManagement.Gateway.Security.Ciphers.TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA': 'false'
      'Microsoft.WindowsAzure.ApiManagement.Gateway.Security.Ciphers.TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA': 'false'
      'Microsoft.WindowsAzure.ApiManagement.Gateway.Security.Ciphers.TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA': 'false'
      'Microsoft.WindowsAzure.ApiManagement.Gateway.Security.Ciphers.TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA': 'false'
      'Microsoft.WindowsAzure.ApiManagement.Gateway.Security.Ciphers.TLS_RSA_WITH_AES_128_CBC_SHA': 'false'
      'Microsoft.WindowsAzure.ApiManagement.Gateway.Security.Ciphers.TLS_RSA_WITH_AES_128_CBC_SHA256': 'false'
      'Microsoft.WindowsAzure.ApiManagement.Gateway.Security.Ciphers.TLS_RSA_WITH_AES_128_GCM_SHA256': 'false'
      'Microsoft.WindowsAzure.ApiManagement.Gateway.Security.Ciphers.TLS_RSA_WITH_AES_256_CBC_SHA': 'false'
      'Microsoft.WindowsAzure.ApiManagement.Gateway.Security.Ciphers.TLS_RSA_WITH_AES_256_CBC_SHA256': 'false'
      'Microsoft.WindowsAzure.ApiManagement.Gateway.Security.Ciphers.TripleDes168': 'false'
    }
    publisherEmail: publisherEmail
    publisherName: publisherName
    publicNetworkAccess: 'Disabled'
    publicIpAddressId: null
    virtualNetworkConfiguration: {
      subnetResourceId: integrationSubnetId
    }
  }
  tags: tags
}

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' existing = {
  name: keyVaultName
}

resource keyVaultRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVaultResourceId, service.name, keyVaultSecretsUserRoleDefinitionId)
  scope: keyVault
  properties: {
    principalId: service.identity.principalId
    roleDefinitionId: keyVaultSecretsUserRoleDefinitionId
    principalType: 'ServicePrincipal'
  }
}

resource backendAuthNamedValue 'Microsoft.ApiManagement/service/namedValues@2024-10-01-preview' = {
  parent: service
  name: backendAuthNamedValueName
  dependsOn: [
    keyVaultRoleAssignment
  ]
  properties: {
    displayName: backendAuthNamedValueName
    keyVault: {
      identityClientId: null
      secretIdentifier: backendAuthSecretUri
    }
    secret: true
  }
}

resource api 'Microsoft.ApiManagement/service/apis@2024-10-01-preview' = {
  parent: service
  name: 'scholar-search-mcp'
  properties: {
    apiType: 'http'
    displayName: 'Scholar Search MCP'
    path: apiPath
    protocols: [
      'https'
    ]
    serviceUrl: backendUrl
    subscriptionRequired: true
  }
}

resource apiOperations 'Microsoft.ApiManagement/service/apis/operations@2024-10-01-preview' = [for operation in operationDefinitions: {
  parent: api
  name: operation.name
  properties: {
    displayName: operation.displayName
    method: operation.method
    urlTemplate: operation.urlTemplate
  }
}]

resource apiPolicy 'Microsoft.ApiManagement/service/apis/policies@2024-10-01-preview' = {
  parent: api
  name: 'policy'
  properties: {
    format: 'xml'
    value: loadTextContent('../policies/scholar-search-policy.xml')
  }
}

resource gatewayPrivateEndpoint 'Microsoft.Network/privateEndpoints@2024-05-01' = {
  name: '${name}-gateway-pe'
  location: location
  properties: {
    privateLinkServiceConnections: [
      {
        name: 'gateway'
        properties: {
          groupIds: [
            'gateway'
          ]
          privateLinkServiceId: service.id
        }
      }
    ]
    subnet: {
      id: privateEndpointSubnetId
    }
  }
  tags: tags
}

resource gatewayPrivateDnsZoneGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2024-05-01' = {
  parent: gatewayPrivateEndpoint
  name: 'default'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'gateway'
        properties: {
          privateDnsZoneId: privateDnsZoneId
        }
      }
    ]
  }
}

output gatewayUrl string = 'https://${service.name}.azure-api.net/${apiPath}'
output serviceName string = service.name
