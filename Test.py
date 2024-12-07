from DiscreteBasicExtraction import ExtractFromExcelDiscrete

ExcelPath = "2024_PersonalityTraits_SurveyData.xlsx"
df3 = ExtractFromExcelDiscrete(ExcelPath ,False ) #The False means you want to use the defualt scaling  (recommended....) 
print(df3.head())


