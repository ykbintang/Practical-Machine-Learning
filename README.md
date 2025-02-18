# Practical-Machine-Learning
Peer-graded Assignment: Prediction Assignment Writeup

Yayes Kasnanda Bintang

2025-02-18

### Objective

The purpose of this project was to quantify how well the participants
performed a barbell lifting exercise and to classify the measurement
read from an accelerometer into 5 different classes (Class A:Class E).

Please reference the links below for the data sources:

<http://groupware.les.inf.puc-rio.br/har>

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

#### Install/load the required packages needed for the creation of the model
    library(caret)
    
    ## Warning: package 'caret' was built under R version 4.3.3    
    ## Loading required package: ggplot2    
    ## Warning: package 'ggplot2' was built under R version 4.3.3    
    ## Loading required package: lattice
  
    library(rpart)
    library(randomForest)
    
    ## Warning: package 'randomForest' was built under R version 4.3.3    
    ## randomForest 4.7-1.2
    
    ## Type rfNews() to see new features/changes/bug fixes.    
    ## 
    ## Attaching package: 'randomForest'
    
    ## The following object is masked from 'package:ggplot2':    
    ## 
    ##     margin
    
#### Load the training and testing datasets
    train<-read.csv("C:/home/project/psw/pml-training.csv",na.strings=c("NA","#DIV/0!",""))
    test<-read.csv("C:/home/project/psw/pml-testing.csv",na.strings=c("NA","#DIV/0!",""))

#### Remove null columns and the first 7 columns that will not be used
    test_clean <- names(test[,colSums(is.na(test)) == 0]) [8:59]
    clean_train<-train[,c(test_clean,"classe")]
    clean_test<-test[,c(test_clean,"problem_id")]

#### Check the dimensions of the clean test and train sets
    dim(clean_test)
    
    ## [1] 20 53
    
    dim(clean_train)
    
    ## [1] 19622    53

#### Split the data into the training and testing datasets
    set.seed(100)
    inTrain<-createDataPartition(clean_train$classe, p=0.7, list=FALSE)
    training<-clean_train[inTrain,]
    testing<-clean_train[-inTrain,]
    dim(training)
    
    ## [1] 13737    53
    
    dim(testing)
    
    ## [1] 5885   53

### Predicting the outcome using 3 different models
#### LDA Model
    lda_model<-train(classe~ ., data=training, method="lda")
    set.seed(200)
    predict<-predict(lda_model,testing)
    confusionMatrix(predict,as.factor(testing$classe))

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1384  179   97   59   51
    ##          B   36  704  111   52  174
    ##          C  138  161  666  117   94
    ##          D  112   36  131  699  106
    ##          E    4   59   21   37  657
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.6984          
    ##                  95% CI : (0.6865, 0.7101)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6181          
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8268   0.6181   0.6491   0.7251   0.6072
    ## Specificity            0.9083   0.9214   0.8950   0.9218   0.9748
    ## Pos Pred Value         0.7819   0.6537   0.5663   0.6448   0.8445
    ## Neg Pred Value         0.9295   0.9095   0.9236   0.9448   0.9168
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2352   0.1196   0.1132   0.1188   0.1116
    ## Detection Prevalence   0.3008   0.1830   0.1998   0.1842   0.1322
    ## Balanced Accuracy      0.8675   0.7697   0.7721   0.8234   0.7910
The LDA model gave a 70% accuracy on the testing set, with the expected out of sample error around 30%.

#### Decision Tree Model
    decision_tree_model<-rpart(classe~ ., data=training,method="class")
    set.seed(300)
    predict<-predict(decision_tree_model,testing,type="class")
    confusionMatrix(predict,as.factor(testing$classe))
    
    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1533  181   22   39    5
    ##          B   59  665  102   97   92
    ##          C   44  153  788   87   79
    ##          D   18   76   78  652  100
    ##          E   20   64   36   89  806
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7551          
    ##                  95% CI : (0.7439, 0.7661)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6897          
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9158   0.5838   0.7680   0.6763   0.7449
    ## Specificity            0.9413   0.9263   0.9253   0.9447   0.9565
    ## Pos Pred Value         0.8612   0.6552   0.6846   0.7056   0.7941
    ## Neg Pred Value         0.9657   0.9027   0.9497   0.9371   0.9433
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2605   0.1130   0.1339   0.1108   0.1370
    ## Detection Prevalence   0.3025   0.1725   0.1956   0.1570   0.1725
    ## Balanced Accuracy      0.9286   0.7550   0.8467   0.8105   0.8507
The Decision Tree Model gave a 74% accuracy on the testing set, with the expected out of sample error around 26%.

#### Random Forest Model
    random_forest_mod<-randomForest(as.factor(classe)~ ., data=training, ntree=500)
    set.seed(300)
    predict<-predict(random_forest_mod, testing, type ="class")
    confusionMatrix(predict,as.factor(testing$classe))
    
    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1671    4    0    0    0
    ##          B    3 1135    4    0    0
    ##          C    0    0 1022   11    2
    ##          D    0    0    0  953    1
    ##          E    0    0    0    0 1079
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9958          
    ##                  95% CI : (0.9937, 0.9972)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9946          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9982   0.9965   0.9961   0.9886   0.9972
    ## Specificity            0.9991   0.9985   0.9973   0.9998   1.0000
    ## Pos Pred Value         0.9976   0.9939   0.9874   0.9990   1.0000
    ## Neg Pred Value         0.9993   0.9992   0.9992   0.9978   0.9994
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2839   0.1929   0.1737   0.1619   0.1833
    ## Detection Prevalence   0.2846   0.1941   0.1759   0.1621   0.1833
    ## Balanced Accuracy      0.9986   0.9975   0.9967   0.9942   0.9986
The Random Forest Model gave a 99.6% accuracy on the testing set, with the expected out of sample error around 0.4%.

### Conclusion
The greatest accuracy was achieved using the Random Forest Model, which gave an accuracy of 99.6%. Hence, this model was further used to make predictions on the exercise performance for 20 participants.

    predict<-predict(random_forest_mod, clean_test, type ="class")
    predict
    
    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E
