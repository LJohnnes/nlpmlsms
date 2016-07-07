# -*- coding: utf-8 -*-
"""
Created on Sun July  3 17:18:09 2016

@author: Rangga Ugahari
"""

import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, NGram
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, NaiveBayes
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.mllib.linalg import Vectors

# Module-level global variables for the `tokenize` function below
PUNCTUATION = set(string.punctuation)
STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

# Function to break text into "tokens", lowercase them, remove punctuation and stopwords, and stem them
def tokenize(text):
    tokens = word_tokenize(text)
    lowercased = [t.lower() for t in tokens]
    no_punctuation = []
    for word in lowercased:
        punct_removed = ''.join([letter for letter in word if not letter in PUNCTUATION])
        no_punctuation.append(punct_removed)
    no_stopwords = [w for w in no_punctuation if not w in STOPWORDS]
    stemmed = [STEMMER.stem(w) for w in no_stopwords]
    return [w for w in stemmed if w]

#DATA PREPARATION
##reading csv file
data = pd.read_csv("sms_spam.csv")
#print(data.head(5))
    
##creating rdd file
sc = SparkContext("local", "app")
sqc = SQLContext(sc)
df = sqc.createDataFrame(data, ['type', 'text'])

#NEW VARIABLE GENERATION
dataCleaned = df.map(lambda x: (1 if x['type'] == 'spam' else 0, tokenize(x['text'])))
dataClean = dataCleaned.map(lambda x: (float(x[0]), x[1]))
dfClean = sqc.createDataFrame(dataClean, ['label', 'words'])
dfClean.show(5)

hashingTF = HashingTF(inputCol="words", outputCol="rawtf-idf", numFeatures=1000)
tf = hashingTF.transform(dfClean)
idf = IDF(inputCol="rawtf-idf", outputCol="features").fit(tf)
dfFinal = idf.transform(tf)

# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(dfFinal)
# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(dfFinal)

# Split the data into training and test sets (20% held out for testing)
(trainingData, testData) = dfFinal.randomSplit([0.8, 0.2])


# Train the model.
#rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
nb = NaiveBayes(smoothing = 1.0, labelCol="indexedLabel", featuresCol="indexedFeatures")

#pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf])
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, nb])
paramGrid = ParamGridBuilder().build()

crossval = CrossValidator(
     estimator=pipeline,
     estimatorParamMaps=paramGrid,
     evaluator=BinaryClassificationEvaluator(),
     numFolds=5)
     
model = crossval.fit(trainingData)

# Compute raw scores on the test set
predictions = model.transform(testData)
predictions.select("prediction", "indexedLabel", "features").show(5)
rddPredictions = predictions.select("prediction", "indexedLabel").rdd
accuracy = rddPredictions.filter(lambda p: (p['prediction'] == p['indexedLabel'])).count() / float(testData.count())
TP = rddPredictions.filter(lambda p: (p['prediction'] == 1 and p['prediction'] == p['indexedLabel'])).count()
TN = rddPredictions.filter(lambda p: (p['prediction'] == 0 and p['prediction'] == p['indexedLabel'])).count()
FP = rddPredictions.filter(lambda p: (p['indexedLabel'] == 1 and p['prediction'] != p['indexedLabel'])).count()
FN = rddPredictions.filter(lambda p: (p['indexedLabel'] == 0 and p['prediction'] != p['indexedLabel'])).count()
print("TP = ", TP)
print("TN = ", TN)
print("FP = ", FP)
print("FN = ", FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)
f1Score = 2*(precision*recall/(precision+recall))
print("Summary Stats")
print('Model Accuracy = ' + str(float(accuracy)))
print("Precision = %s" % precision)
print("Recall = %s" % recall)
print("F1 Score = %s" % f1Score )