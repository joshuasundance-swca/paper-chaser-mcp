param acrResourceId string
param keyVaultResourceId string
param principalId string
param keyVaultPrincipalIds array = [
  principalId
]

var acrName = last(split(acrResourceId, '/'))
var keyVaultName = last(split(keyVaultResourceId, '/'))

var acrPullRoleDefinitionId = subscriptionResourceId(
  'Microsoft.Authorization/roleDefinitions',
  '7f951dda-4ed3-4680-a7ca-43fe172d538d'
)
var keyVaultSecretsUserRoleDefinitionId = subscriptionResourceId(
  'Microsoft.Authorization/roleDefinitions',
  '4633458b-17de-408a-b874-0445c86b69e6'
)

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: acrName
}

resource vault 'Microsoft.KeyVault/vaults@2023-07-01' existing = {
  name: keyVaultName
}

resource acrRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acrResourceId, principalId, acrPullRoleDefinitionId)
  scope: acr
  properties: {
    principalId: principalId
    roleDefinitionId: acrPullRoleDefinitionId
    principalType: 'ServicePrincipal'
  }
}

resource keyVaultRoleAssignments 'Microsoft.Authorization/roleAssignments@2022-04-01' = [
  for keyVaultPrincipalId in keyVaultPrincipalIds: {
    name: guid(keyVaultResourceId, keyVaultPrincipalId, keyVaultSecretsUserRoleDefinitionId)
    scope: vault
    properties: {
      principalId: keyVaultPrincipalId
      roleDefinitionId: keyVaultSecretsUserRoleDefinitionId
      principalType: 'ServicePrincipal'
    }
  }
]

output acrRoleAssignmentId string = acrRoleAssignment.id
output keyVaultRoleAssignmentIds array = [
  for (keyVaultPrincipalId, index) in keyVaultPrincipalIds: keyVaultRoleAssignments[index].id
]
