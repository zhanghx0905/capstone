library(tidyverse)
library(text2vec)
library(e1071)
library(kernlab)
library(caret)
library(randomForest)
library(naivebayes)
library(tidytext)
library(jiebaR)
library(glmnet)
library(parallel)
wk<-worker()



text.train<-read_tsv('../data/train.txt',col_names = c('txt','type'))
text.test<-read_tsv('../data/test.txt',col_names = c('txt','type'))
text.dev<-read_tsv('../data/dev.txt',col_names = c('txt','type'))

####预处理，创建dtm####
train_tokens<-word_tokenizer(c(text.train$txt,text.dev$txt))
it_train <- itoken(train_tokens)
vocab <- create_vocabulary(it_train)
vectorizer <- vocab_vectorizer(vocab)
dtm_train <- create_dtm(it_train, vectorizer)


####Naive Bayes####

type<-as.factor(text.train$type)
model.nb<-multinomial_naive_bayes(y = type,x = dtm_train[1:180000,] )
ypre.nb<-predict(model.nb,newdata = dtm_train[180001:190000,])
confusionMatrix(as.factor(text.dev$type),ypre.nb)#0.864



####tf-idf方法函数####

search_word<-function(word,dictionary){
  b<-integer(0)
  a<-which(dictionary$words==word)
  c<-rep(0,10)
  if(setequal(a,b)){
    return(c)
  }
  else{
    a.l<-length(a)
    if(a.l==1){
      n<-dictionary$type[a]
      d<-dictionary$tf_idf[a]
      c[n+1]<-d
      return(c)
    }
    else{
      c.list<-map(a,function(x){n<-dictionary$type[x];
      d<-dictionary$tf_idf[x];c<-rep(0,10)
      c[n+1]<-d;
      return(c)})
      c.mat<-matrix(unlist(c.list),a.l,10,byrow = T)
      return(colSums(c.mat))
      
    }
  }
}

sentense_judge<-function(sentence,dictionary){
  a<-rep(0,9)
  wv<-segment(sentence,jiebar = wk)
  wv.l<-length(wv)
  w.list<-map(wv,search_word,dictionary = dictionary)
  w.mat<-matrix(unlist(w.list),wv.l,10,byrow = T)
  w.mat.v<-colSums(w.mat)
  return(which.max(w.mat.v)-1)
}

######构建tf-idf########

text.train%>% 
  mutate(words = map(text.train$txt,segment,jieba = wk)) %>% 
  select(type,words) ->text.train1

text.train1 %>% 
  unnest() %>% 
  count(type,words) -> f_table

f_table %>%
  bind_tf_idf(term = words,document = type,n = n) -> tf_idf_table

##########字典查询方法##########

type.pre<-Vectorize(sentense_judge,vectorize.args = 'sentence')


tf.pre<-type.pre(text.dev$txt,tf_idf_table)
confusionMatrix(as.factor(tf.pre),as.factor(text.dev$type))

#############tf-idf+传统统计学习方法###################

cl <- makeCluster(4)

clusterEvalQ(cl, source("func.r"))

vec.mat<-parLapply(cl,1:180000,function(x){sentense_judge(text.train$txt[x],tf_idf_table)})%>%unlist%>%matrix(nrow = 180000,ncol = 10,byrow = T)

dev.mat<-parLapply(cl,1:10000,function(x){sentense_judge(text.dev$txt[x],tf_idf_table)})%>%unlist%>%matrix(nrow = 10000,ncol = 10,byrow = T)

train.log<-glmnet(y = text.train$type, x = as.matrix(text.train[,-1]),family = 'multinomial')
dev.pre<-predict(train.log,as.matrix(text.dev[,-1]),type = 'class')
confusionMatrix(text.dev$type,as.factor(dev.pre[,60]))#82.4%


train.rf<-randomForest( type~.,text = text.train)
dev.pre<-predict(train.rf,text.dev[,-1])
confusionMatrix(text.dev$type,as.factor(dev.pre))#86.4

train.svm<-ksvm( type~.,text = text.train)
dev.pre<-predict(train.svm,text.dev[,-1])
confusionMatrix(text.dev$type,as.factor(dev.pre))#81.8
