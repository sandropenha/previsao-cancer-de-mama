#### <- TOPICO
## <- Subtopico
# <-  comentário

------------------------------------------------------------------------------

#### SOBRE ####
# PREVISAO DE CANCER DE MAMA
# O cancêr de mama, é atualmente um dos tipos de cancer mais mortais.
# Iremos investigar a utilidade da aprendizagem de máquina para detectção de tumores malignos ou benignos, 
# aplicando técnicas de machine learning e data science utilizando do repositório de Machine Learning da UCI
# http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names
# Os dados presentes no datase tratam-se de dados reais.

#### DIRETÓRIO DE TRABALHO ####
setwd('E:/projetos/previsao_cancer_de_mama');getwd()

##### LIBRARYS ####
pacman::p_load(tidyverse,class,caTools,gmodels,DMwR,caret,e1071)

#### DATASET ####
df <- read.csv("dataset.csv")
head(df);glimpse(df)

##  Ajustando os dados:
df <- df %>% select(-id) %>% 
  mutate(diagnosis = factor(diagnosis, levels = c("B","M"), labels = c(0,1))) %>% 
  mutate_if(is.numeric,scale)

## Verificando a proporção:
round(prop.table(table(df$diagnosis))*100,2)

## Aplicando balanceamento:
df2 <- DMwR::SMOTE(diagnosis~.,df, perc.over = 900,perc.under = 100)
round(prop.table(table(df2$diagnosis))*100,2)

## Dividindo dados de treino e de teste:
treino <- df2 %>% sample_frac(0.7)
teste <- df2 %>% sample_frac(0.3)

## Verificando balanceamento nos dados de treino e de teste:
round(prop.table(table(treino$diagnosis))*100,2)
round(prop.table(table(teste$diagnosis))*100,2)

#### CONSTRUINDO O MODELO COM KNN ####
set.seed(400)

## Arquivo de controle:
ctrl <- trainControl(method = "repeatedcv", repeats = 3)

## Modelo:
knn_v1 <- caret::train(diagnosis~.,
                       data = treino,
                       method = "knn",
                       trControl = ctrl,
                       tuneLength = 20);knn_v1

plot(knn_v1)

## Previsões:
previsao_knn <- predict(knn_v1,newdata=teste);previsao_knn
confusionMatrix(previsao_knn,teste$diagnosis)

# 0,9934 de acuracia;
# onde era 0, acertou 569 e errou 1
# onde era 1, acertou 631 e errou 7


## Modelo knn com métrica ROC:
controle <- trainControl(method = "repeatedcv",
                         repeats =3,
                         classProbs = TRUE,
                         summaryFunction = twoClassSummary)

df3 <- df2 %>% mutate(diagnosis = factor(diagnosis,levels = c("0","1"), labels = c("Benigno","Maligno")))
treino2 <- df3 %>%  sample_frac(0.7)
teste2 <- df3 %>% sample_frac(0.3)

knn_v2 <- caret::train(diagnosis~.,
                       data = treino2,
                       method = "knn",
                       trControl = controle,
                       metric = "ROC",
                       tuneLength = 20);knn_v2

plot(knn_v2)

## Previsões:
previsao_knn2 <- predict(knn_v2,teste2[-1]);previsao_knn2
confusionMatrix(previsao_knn2,teste2$diagnosis)

# 1 de acuracia
# onde era 0 acertou 574 e errou 0
# onde era 1 acertou 634 e errou 0


#### MODELO UTILIZANDO O CLASSIFICADOR NAIVE BAYES ####
modelo_naive <- naiveBayes(x = treino[-1],y=treino$diagnosis);print(classificador)
previsao <- predict(modelo_naive,teste[-1])
conf.matrix <- table(teste[,1],previsao)
confusionMatrix(conf.matrix)

# 94% de acurácia
# onde era 0 acertou 559 e errou 17
# onde era 1 acertou 582 e errou 50


#### MODELO UTILIZANDO REGRESSÃO LOGISTICA ####
control <- trainControl(method = "repeatedcv", number = 10, repeats = 2)
modelo_reg_log <- train(diagnosis~., data = treino, method = "glm", trControl = control)

## Visualizando variáveis mais importantes:
importance <- varImp(modelo_reg_log, scale = FALSE);plot(importance)

## Fazendo previsoes:
previsoes <- predict(modelo_reg_log, teste[-1])

## Avaliando o modelo:
confusionMatrix(table(data = previsoes, reference = teste[,1]), positive = "1")

# 1 de acurácia
# onde era 0 acertou 576 e errou 0
# onde era 1 acertou 632 e errou 0



