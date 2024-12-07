from DiscreteBasicExtraction import ExtractFromExcelDiscrete
from PCA_Methods import ElbowBestCluster,BicsBestCluster,PCA_PlotClusters
from AutoEncoder import DimentionalityReductionCluster

ExcelPath = "2024_PersonalityTraits_SurveyData.xlsx"
df3 = ExtractFromExcelDiscrete(ExcelPath ,False ) #The False means you want to use the defualt scaling  (recommended....) 
print(df3.head())

ElbowBestCluster(df3) #Visual Aid to decide
BicsBestCluster(df3) #Visual Aid to decide
PCA_PlotClusters(df3, 3) 
PCA_PlotClusters(df3, 5)
IdealNumberOfCluster = PCA_PlotClusters(df3, 10) 
print(IdealNumberOfCluster)
DimentionalityReductionCluster(df3)


