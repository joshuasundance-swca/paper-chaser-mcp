param location string
param name string
param privateDnsZoneId string
param privateEndpointSubnetId string
param tags object = {}

resource registry 'Microsoft.ContainerRegistry/registries@2025-11-01' = {
  name: name
  location: location
  sku: {
    name: 'Premium'
  }
  properties: {
    adminUserEnabled: false
    anonymousPullEnabled: false
    dataEndpointEnabled: false
    networkRuleBypassOptions: 'None'
    policies: {
      exportPolicy: {
        status: 'disabled'
      }
      retentionPolicy: {
        days: 7
        status: 'enabled'
      }
    }
    publicNetworkAccess: 'Disabled'
  }
  tags: tags
}

resource registryPrivateEndpoint 'Microsoft.Network/privateEndpoints@2024-05-01' = {
  name: '${name}-pe'
  location: location
  properties: {
    privateLinkServiceConnections: [
      {
        name: 'registry'
        properties: {
          groupIds: [
            'registry'
          ]
          privateLinkServiceId: registry.id
        }
      }
    ]
    subnet: {
      id: privateEndpointSubnetId
    }
  }
  tags: tags
}

resource registryPrivateDnsZoneGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2024-05-01' = {
  parent: registryPrivateEndpoint
  name: 'default'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'registry'
        properties: {
          privateDnsZoneId: privateDnsZoneId
        }
      }
    ]
  }
}

output loginServer string = registry.properties.loginServer
output resourceId string = registry.id
