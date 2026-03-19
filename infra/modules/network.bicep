param environmentName string
param location string
param name string
param tags object = {}

var environmentOctet = environmentName == 'prod'
  ? 43
  : environmentName == 'staging'
    ? 42
    : environmentName == 'dev'
      ? 41
      : 44

var virtualNetworkName = 'vnet-${name}'
var containerAppsSubnetName = 'snet-container-apps'
var apiManagementSubnetName = 'snet-apim'
var privateEndpointSubnetName = 'snet-private-endpoints'

var virtualNetworkAddressPrefix = '10.${environmentOctet}.0.0/16'
var containerAppsSubnetAddressPrefix = '10.${environmentOctet}.0.0/23'
var apiManagementSubnetAddressPrefix = '10.${environmentOctet}.2.0/27'
var privateEndpointSubnetAddressPrefix = '10.${environmentOctet}.3.0/24'

var privateDnsZoneNames = [
  'privatelink.azure-api.net'
  'privatelink.azurecr.io'
  'privatelink.vaultcore.azure.net'
  'privatelink.${location}.azurecontainerapps.io'
]

resource apiManagementNetworkSecurityGroup 'Microsoft.Network/networkSecurityGroups@2024-05-01' = {
  name: 'nsg-${name}-apim'
  location: location
  tags: tags
}

resource virtualNetwork 'Microsoft.Network/virtualNetworks@2024-05-01' = {
  name: virtualNetworkName
  location: location
  properties: {
    addressSpace: {
      addressPrefixes: [
        virtualNetworkAddressPrefix
      ]
    }
    subnets: [
      {
        name: containerAppsSubnetName
        properties: {
          addressPrefix: containerAppsSubnetAddressPrefix
          delegations: [
            {
              name: 'containerAppsDelegation'
              properties: {
                serviceName: 'Microsoft.App/environments'
              }
            }
          ]
        }
      }
      {
        name: apiManagementSubnetName
        properties: {
          addressPrefix: apiManagementSubnetAddressPrefix
          delegations: [
            {
              name: 'apiManagementDelegation'
              properties: {
                serviceName: 'Microsoft.Web/serverFarms'
              }
            }
          ]
          networkSecurityGroup: {
            id: apiManagementNetworkSecurityGroup.id
          }
        }
      }
      {
        name: privateEndpointSubnetName
        properties: {
          addressPrefix: privateEndpointSubnetAddressPrefix
          privateEndpointNetworkPolicies: 'Disabled'
        }
      }
    ]
  }
  tags: tags
}

resource privateDnsZones 'Microsoft.Network/privateDnsZones@2024-06-01' = [
  for zoneName in privateDnsZoneNames: {
    name: zoneName
    location: 'global'
    tags: tags
  }
]

resource privateDnsZoneLinks 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2024-06-01' = [
  for (zoneName, index) in privateDnsZoneNames: {
    parent: privateDnsZones[index]
    name: '${name}-link'
    location: 'global'
    properties: {
      registrationEnabled: false
      virtualNetwork: {
        id: virtualNetwork.id
      }
    }
  }
]

output apiManagementPrivateDnsZoneId string = privateDnsZones[0].id
output acrPrivateDnsZoneId string = privateDnsZones[1].id
output keyVaultPrivateDnsZoneId string = privateDnsZones[2].id
output containerAppsPrivateDnsZoneId string = privateDnsZones[3].id
output apiManagementSubnetId string = resourceId(
  'Microsoft.Network/virtualNetworks/subnets',
  virtualNetwork.name,
  apiManagementSubnetName
)
output containerAppsInfrastructureSubnetId string = resourceId(
  'Microsoft.Network/virtualNetworks/subnets',
  virtualNetwork.name,
  containerAppsSubnetName
)
output privateEndpointSubnetId string = resourceId(
  'Microsoft.Network/virtualNetworks/subnets',
  virtualNetwork.name,
  privateEndpointSubnetName
)
output virtualNetworkId string = virtualNetwork.id
