library(warbleR)
library(readr)
library(tuneR)
sample <- function(audio){
  audio_path<-paste('C:/Users/tejas/Downloads',audio,sep="")
  train_audio<-readWave(audio_path)
  str(train_audio)
  dataframe = data.frame(audio,2,1,20)
  names(dataframe)<-c("sound.files","selec","start","end")
  a<-specan(X = dataframe,bp=c(0.09,0.25),path = 'C:/Users/tejas/Downloads')
  View(a)
  return (a)
}