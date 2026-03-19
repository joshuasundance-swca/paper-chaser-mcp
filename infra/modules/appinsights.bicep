param location string
param name string
param workspaceResourceId string
param tags object = {}

resource applicationInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: name
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    DisableLocalAuth: true
    Flow_Type: 'Bluefield'
    IngestionMode: 'LogAnalytics'
    WorkspaceResourceId: workspaceResourceId
  }
  tags: tags
}

output connectionString string = applicationInsights.properties.ConnectionString
output resourceId string = applicationInsights.id
