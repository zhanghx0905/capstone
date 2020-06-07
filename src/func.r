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

