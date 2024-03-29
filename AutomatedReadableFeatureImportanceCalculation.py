import time
start_time = time.time()

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.types import *
from coalesce import coalesce
from pyspark.sql.functions import col
from pyspark.ml.feature import *
from pyspark.ml.regression import RandomForestRegressor, DecisionTreeRegressor
from pyspark.ml import *
import time
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.storagelevel import StorageLevel
import time
import sys                                                                                                                                                        
reload(sys)                                                                                                                                                       
sys.setdefaultencoding('utf8') 

print("------ CONFIG -------")


#CONF CLUSTER
conf=SparkConf()\
.set('spark.network.timeout','5000000s')\
.set('spark.executor.heartbeatInterval','4500000s')
sc = SparkContext(conf=conf)
#sc.setCheckpointDir("checkpointdir2")
sc.setCheckpointDir("checkpointdir2")
sqlContext=SQLContext(sc)


print("------ reading data -------")
data=sqlContext.read.parquet("data.parquet").sample(0.5)


print("")
print(sc.uiWebUrl)

print("--------- factorization of categorical values ------------")

features=['seq','dow','first_dow','code_evt','ss_code_evt','libelle_agence','libelle_dro','libelle_agence_premier_evt_transport','libelle_dro_premier_evt_transport','agence_destination','v_code_premierevttransport_reel','code_service','lieu_evt','code_produit','code_postal_destinataire','code_postal_expediteur','v_socode','v_sum_mutables','v_sum_non_mutables','v_lieu_premierevttransport_reel','first_code_evt','first_lieu_evt','time','contractual_time_difference']
df=data[features]
df.show(1)
print(df.dtypes)

print("------ train test split --------")
(trainingData,testData)=data.randomSplit([0.7,0.3])


#############################################################################################
print("---------------- STARTING AUTOMATIC PIPELINE CREATION ------------------------")######
#############################################################################################

catNames=[]
catNames_index=[]
catValIndexed=[]

for f in features:
	if df.schema[f].dataType == StringType():
		print(df.schema[f].name)
		catNames.append(f)
		t=f+"_index"
		catNames_index.append(t)
		print(t)
		catValIndexed.append(StringIndexer(inputCol=f,outputCol=t).setHandleInvalid("keep"))
		cat=dict(zip(catNames,catValIndexed))
	else:
		print(f+" has not a well defined type or is a Double or String")

integ=['seq']
inputColsOhe=integ+catNames_index
print("inputColsOhe")
print(inputColsOhe)

outputColsOhe=[]
for a in inputColsOhe:
	if a.endswith('_index'):
		outputColsOhe.append(a[:-6]+'_category')
	else:
		outputColsOhe.append(a+'_category')

ohe = OneHotEncoderEstimator(inputCols=inputColsOhe,outputCols=outputColsOhe).setHandleInvalid("keep")


numeric=[]
for f in featTry:
	if df.schema[f].dataType == DoubleType() and df.schema[f].name not in integ:
		numeric.append(df.schema[f].name)
numericalAssembler=VectorAssembler(inputCols=numeric,outputCol='numerical_features')

numericalScaler=MinMaxScaler(inputCol='numerical_features',outputCol='scaled_numerical_features')

inputColsAssembler=outputColsOhe
inputColsAssembler.append('scaled_numerical_features')
assembler= VectorAssembler(inputCols=inputColsAssembler,outputCol='features')

labelCol="label"

print("--------------- Random Forest Model definition ----------------")
rf=RandomForestRegressor(labelCol=labelCol, featuresCol='features', predictionCol='prediction', maxBins=100, numTrees=100, maxDepth=11, subsamplingRate=0.1)

print("------------- transformer pipeline --------------")
steps=[ohe,numericalAssembler,numericalScaler,assembler]
for a in steps:
	catValIndexed.append(a)
transformerStages=catValIndexed
transformerPipeline=Pipeline(stages=transformerStages)
transformer=transformerPipeline.fit(trainingData)
transformedTrainingData=transformer.transform(trainingData)

print("------------ REGRESSION MODEL ---------------")
modelPipeline=Pipeline(stages=[rf])
print("------------------------------------------ modelPipeline.fit(test).persist(StorageLevel.DISK_ONLY) ------------------------------------------------")
model=modelPipeline.fit(transformedTrainingData)



#############################################################################################
print("---------------- STARTING FEATURE IMPORTANCE AGREGATION ------------------------")####
#############################################################################################

import pandas as pd
pandasDF=pd.DataFrame(transformedTrainingData.schema["features"].metadata["ml_attr"]["attrs"]["numeric"]+transformedTrainingData.schema["features"].metadata["ml_attr"]["attrs"]["binary"]).sort_values("idx")
pandasDF['importance']=model.stages[-1].featureImportances.toArray()
features=trainingData.schema.names

def getCategorical(df):
	tab=[]
	for a in df['name']:
		if 'category' in a:
			tab.append(a.split('category'))
	tab2=[]
	for a in tab:
		tab2.append((a[0]+'category').encode("utf-8"))
	t=set(tab2)
	new=list(t) 
	return(new)

cat=(getCategorical(pandasDF))

def hasNumbers(inputString):
	return any(char.isdigit() for char in inputString)
'''
for a in pandasDF["name"]:
	if 'numerical' in a:
		print(a)
'''

def getNumerical(df):
	b=[]
	for a in df['name']:
		if 'numerical' in a:
			b.append(a.replace("/\D/g","") if hasNumbers(a) else b.append(a))
			print(a)
			print(b)
	tab2=[]
	for c in b:
		tab2.append(str(c).encode('utf-8'))
	t=set(tab2)
	new=list(t)
	print(new)
	if 'None' in new:
		new.remove('None') 
	return (new)


num=getNumerical(pandasDF)


print("--------- begin last step ---------------")
feat=[]
imp=[0.0 for x in range(len(cat))]
print(imp)
idx=0
for i in (range(0,len(cat))):
	for j in range(0,len(pandasDF)):
		print("cat[",i,"]")
		print(cat[i])
		print("pandasDF[name][",j,"]")
		print(pandasDF["name"][j])
		#if cat[i] in pandasDF["name"][j]:
		#if cat[i] in pandasDF["name"][j] and cat[i] not in feat and pandasDF["name"][j].startswith(cat[i]):
		if cat[i] in pandasDF["name"][j] and pandasDF["name"][j].startswith(cat[i]):
			if cat[i] not in feat:
				print("-----------passed 1st condition-----------")
				feat.append(cat[i])
				print(feat)
				imp[i]=imp[i]+pandasDF["importance"][j]
				print(pandasDF["importance"][j])
				print(imp)
			else: 
				print("-----------passed 2nd condition-----------")
				print(feat)
				imp[i]=imp[i]+pandasDF["importance"][j]
				print(pandasDF["importance"][j])
				print(imp)

print(feat)
print(imp)

# contains prints to check that the process is well behaving
featNum=[]
impNum=[0.0 for x in range(len(num))]
for i in (range(0,len(num))):
	for j in range(0,len(pandasDF)):
		print("num[",i,"]")
		print(num[i])
		print("pandasDF[name][",j,"]")
		print(pandasDF["name"][j])
		#if cat[i] in pandasDF["name"][j]:
		#if cat[i] in pandasDF["name"][j] and cat[i] not in feat and pandasDF["name"][j].startswith(cat[i]):
		if num[i] in pandasDF["name"][j] and pandasDF["name"][j].startswith(num[i]):
			if num[i] not in featNum:
				print("-----------passed 1st condition-----------")
				featNum.append(num[i])
				print(featNum)
				impNum[i]=impNum[i]+pandasDF["importance"][j]
				print(pandasDF["importance"][j])
				print(impNum)
			else: 
				print("-----------passed 2nd condition-----------")
				print(featNum)
				impNum[i]=impNum[i]+pandasDF["importance"][j]
				print(pandasDF["importance"][j])
				print(impNum)

time=0.0
ctime=0;0
for i in (range(0,len(featNum))):
	if '_time' in featNum[i] or '_0' in featNum[i]:
		time=time+impNum[i]
	else:
		ctime=ctime+impNum[i]
numeric=['time','contractual_time_difference']
numVal=[time,ctime]

features=feat+numeric
importances=imp+numVal

print(feat)
print(imp)

print("len(feat)")
print(len(feat))
print("len(imp)")
print(len(imp))
print("len(cat)")
print(len(cat))
print("idx")
print(idx)

print(cat)
print(num)

print("")
print("")
print("")

for i in (range(0,len(features))):
	if importances[i]>0.01:
		print(features[i]," : ", importances[i])
print(sum(importances))


