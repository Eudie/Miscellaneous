## Author Eudie
# For MoonFrog Labs

# rm(list=ls())
# setwd("~/PythonProjects/MoonFrog")

# Libraries----
  library(ggplot2)
  library(plyr)

## Getting Data----
  # Making dummy data (provided by MoonFrog Labs) for this exercise, but we can get data from any source
  Date <- c("01/06/16","01/06/16","01/06/16","02/06/16","02/06/16","02/06/16","02/06/16","03/06/16","03/06/16","03/06/16","04/06/16","04/06/16","05/06/16","05/06/16","05/06/16","05/06/16") 
  Source <- c("S1", "S3", "S2", "S4", "S2", "S3", "S1", "S5", "S2", "S3", "S1", "S2", "S6", "S2", "S3", "S4")
  Inflow <- c(271, 368, 425, 580, 233, 243, 428, 164, 461, 180, 258, 153, 443, 496, 476, 305)
  
  df <- data.frame(Date, Source, Inflow)

df$Date <- as.Date(df$Date, "%d/%m/%y")

## Funtion used----
# Following function takes input a raw data frame and restructure it as required to make
# Asumption: Date on which there is no Sorce-Inflow, that combination is not populate in the table
# For eg. if, On "4-June" there is no Inflow from "S3" than there will be no row for Date = "4-June" and Source = "S3"
# To make chart of continues date and for all sources restructuring is necessary
restructure <- function(dataframe){
  # To get continuous dates
  cont_dates <- seq(min(dataframe$Date), max(dataframe$Date), by="days")
  
  # To change order of source dynamically
  temp <- as.data.frame(aggregate(dataframe$Inflow, by=list(Source=dataframe$Source), FUN=sum))
  temp$Source <- factor(temp$Source , levels=temp$Source[order(temp$x, decreasing = TRUE)])
  
  # All possible combination of Date and Source
  output <- as.data.frame(expand.grid(cont_dates, temp$Source, KEEP.OUT.ATTRS = FALSE))
  colnames(output) <- c("Date", "Source")
  
  # Refilling the Inflow value
  output <- merge(output, dataframe, by= c("Date", "Source"), all.x = TRUE)
  output$Inflow[is.na(output$Inflow)] <- 0
  
  return(output)
}

## Main----
  final_df <- restructure(df)
  ggplot(final_df, aes(x=  Date,y =  Inflow)) + geom_area(aes(colour = Source, fill= Source), position = 'stack', alpha = 0.8) + 
    scale_x_date() + labs(title = "Time series of Inflow by Source")
  ggsave("Time Series of Inflow by Source.jpeg", width = 11.25, height = 7.5)

##End----
