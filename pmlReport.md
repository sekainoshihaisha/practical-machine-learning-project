---
title: "Human activity recognition report"
author: "Brian Téllez"
date: "9/12/2020"
output: 
    html_document:
        keep_md: true
---



## Summary

Mechanical movement of some parts of the body were recorded by different artifacts to quantified those movements according to different parameters. In this experiment, accelerometers were reading lectures of belt, forearm, arm and dumbell. The purpose of this study is to give an overview of how well the subjects did with the experiment.

## Data analysis

First it is necessay to load the needed packages for the analysis and models that will be used.


```r
library(caret)
library(rattle)
```

One of the basic things that is needed to do is to load the data.


```r
testingFile <- read.csv("data/pml-testing.csv")
trainingFile <- read.csv("data/pml-training.csv")
```

One of the approaches to know the data is to look at the dimensions of the dataset.


```r
dim(testingFile)
```

```
## [1]  20 160
```


```r
dim(trainingFile)
```

```
## [1] 19622   160
```

As it can be seen:

* The testing file has 160 variables and 20 observations.

* The training file has 160 variables and 19622 observations.

In this aspect, it can be a little bit overwhelming to get a peek in the data, since there is a large number of variables in the two datasets.

One of the considerations of large datasets is that there may be some null and/or non beneficial data in the observations. In this case, it is needed to clean the data. For this instance, variables with little variance and variables with a considerable quantity of missing values should be listed out.


```r
noVarianceData <- nearZeroVar(trainingFile)
trainingFile <- trainingFile[, -noVarianceData]

naVariables <- sapply(trainingFile, function(x) mean(is.na(x))) > 0.95
trainingFile <- trainingFile[, naVariables == FALSE]
```

There also are some variables that are not part of the valuable data, so the dataset can be reduced a little bit more in that aspect.


```r
trainingFile <- trainingFile[, -c(1:7)]
dim(trainingFile)
```

```
## [1] 19622    52
```

As shown above, the variables were reduced by 67.5%. With this resulting dataset, it can be said that it is ready to get some models applied.

For this part of the analysis, the train data should be split into two to get the prediction work in the process. This analysis may be taken before considering the 20 principals cases in the testing archive.


```r
inTrain <- createDataPartition(trainingFile$classe, p = 0.7, list = FALSE)
training <- trainingFile[inTrain,]
testing <- trainingFile[-inTrain,]
```

With these data established, some models can be stated to get the most trustful for the testing observations of the original data.

### Linear discriminant analysis


```r
ldaModFit <- train(classe ~ ., data = training, method = "lda")
```


```r
ldaPrediction <- predict(ldaModFit, newdata = testing)
confusionMatrix(ldaPrediction, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1334  164   94   46   37
##          B   37  740  104   54  198
##          C  150  141  682  107  105
##          D  146   48  121  700  126
##          E    7   46   25   57  616
## 
## Overall Statistics
##                                         
##                Accuracy : 0.6919        
##                  95% CI : (0.68, 0.7037)
##     No Information Rate : 0.2845        
##     P-Value [Acc > NIR] : < 2.2e-16     
##                                         
##                   Kappa : 0.6108        
##                                         
##  Mcnemar's Test P-Value : < 2.2e-16     
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7969   0.6497   0.6647   0.7261   0.5693
## Specificity            0.9190   0.9172   0.8965   0.9104   0.9719
## Pos Pred Value         0.7964   0.6531   0.5755   0.6135   0.8202
## Neg Pred Value         0.9192   0.9160   0.9268   0.9444   0.9092
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2267   0.1257   0.1159   0.1189   0.1047
## Detection Prevalence   0.2846   0.1925   0.2014   0.1939   0.1276
## Balanced Accuracy      0.8580   0.7834   0.7806   0.8183   0.7706
```

As shown above, there is an aproximate accuracy of 70%, given that there is an error output of 30%. This may lead to a big breach in what it is expected, meaning that using this model may give some volatile results.

### Bayesian method


```r
nbModFit <- train(classe ~ ., method = "nb", data = training)
```


```r
nbPrediction <- predict(nbModFit, newdata = testing)
confusionMatrix(nbPrediction, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1463  239  240  177   57
##          B   37  754   67    3   99
##          C   51   84  683  108   40
##          D  111   54   33  626   45
##          E   12    8    3   50  841
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7421          
##                  95% CI : (0.7307, 0.7532)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6701          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8740   0.6620   0.6657   0.6494   0.7773
## Specificity            0.8307   0.9566   0.9418   0.9506   0.9848
## Pos Pred Value         0.6723   0.7854   0.7070   0.7204   0.9201
## Neg Pred Value         0.9431   0.9218   0.9303   0.9326   0.9515
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2486   0.1281   0.1161   0.1064   0.1429
## Detection Prevalence   0.3698   0.1631   0.1641   0.1477   0.1553
## Balanced Accuracy      0.8523   0.8093   0.8037   0.8000   0.8810
```

For the bayesian method, the accuracy is an approximate of 75% (a little bit trusthful than the linear discriminant analysis), setting with an estimated error of 25%. Eventhough the breach of accuracy is lower than the LDA model, it may cause some noise in the analysis.

### Decision tree


```r
dtModFit <- train(classe ~ ., method = "rpart", data = training)
fancyRpartPlot(dtModFit$finalModel)
```

![](pmlReport_files/figure-html/unnamed-chunk-12-1.png)<!-- -->


```r
dtPrediction <- predict(dtModFit, newdata = testing)
confusionMatrix(dtPrediction, as.factor(testing$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1019  183   29   70   26
##          B  230  672  133  161  344
##          C  334  201  848  275  292
##          D   90   83   16  458  110
##          E    1    0    0    0  310
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5619          
##                  95% CI : (0.5491, 0.5747)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.4501          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.6087   0.5900   0.8265  0.47510  0.28651
## Specificity            0.9269   0.8171   0.7732  0.93924  0.99979
## Pos Pred Value         0.7679   0.4364   0.4349  0.60502  0.99678
## Neg Pred Value         0.8563   0.8925   0.9548  0.90133  0.86150
## Prevalence             0.2845   0.1935   0.1743  0.16381  0.18386
## Detection Rate         0.1732   0.1142   0.1441  0.07782  0.05268
## Detection Prevalence   0.2255   0.2617   0.3314  0.12863  0.05285
## Balanced Accuracy      0.7678   0.7036   0.7999  0.70717  0.64315
```

The decision tree is one of the lowest accuray model thus far. With an approximate accuracy level of 60% it would leave with a 40% estimate error level. The use of this model would be poor for the study with the give situation of having some proper considerations for the dataset.

### Random forest


```r
rfControl <- trainControl(method = "cv", number = 3, verboseIter = FALSE)
rfModFit <- train(classe ~ ., method = "rf", trControl = rfControl, data = training)
plot(rfModFit$finalModel, main = "")
```

![](pmlReport_files/figure-html/unnamed-chunk-14-1.png)<!-- -->


```r
rfPrediction <- predict(rfModFit, newdata = testing)
confusionMatrix(rfPrediction, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1670   10    0    0    0
##          B    3 1126    7    0    0
##          C    1    2 1017    8    2
##          D    0    1    2  955    1
##          E    0    0    0    1 1079
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9935          
##                  95% CI : (0.9911, 0.9954)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9918          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9976   0.9886   0.9912   0.9907   0.9972
## Specificity            0.9976   0.9979   0.9973   0.9992   0.9998
## Pos Pred Value         0.9940   0.9912   0.9874   0.9958   0.9991
## Neg Pred Value         0.9990   0.9973   0.9981   0.9982   0.9994
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2838   0.1913   0.1728   0.1623   0.1833
## Detection Prevalence   0.2855   0.1930   0.1750   0.1630   0.1835
## Balanced Accuracy      0.9976   0.9932   0.9943   0.9949   0.9985
```

The random forest has the highest rate of accuracy of >99%, leaving with a <1% estimate error. This indicates that the random forest would be the ideal model instance for this study. In this instance, the RF model will be the one that has to be used for the real testing dataset to get a more realistic output according to the available data.

## Final results

As what it can be considered in the models tested for the training dataset, the accuracies from each one would be the exact data needed to stablish the reasonable method to use.

In this case, modeling with random forest would give a close approach to the expected output for the original testing file, since it is the model with an accuracy near to 100%.


```r
predict(rfModFit, newdata = testingFile)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
