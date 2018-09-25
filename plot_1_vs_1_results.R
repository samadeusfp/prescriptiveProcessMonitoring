library(ggplot2)
library(plyr)
library(RColorBrewer)
library(reshape2)

options(digits=15) 


setwd("/Users/stephanf/SCP/randomSplitResults")
files <- list.files()
data <- data.frame()
for (file in files) {
  tmp <- read.table(file, sep=";", header=T)
  tmp$cls <- "rf"
  data <- rbind(data, tmp)
}

data$metric <- gsub("_mean", "", data$metric)
data$ratio <- paste(data$c_miss, data$c_action, sep=":")

base_size <- 36
line_size <- 1
point_size <- 4

data <- subset(data, dataset != "bpic2018")
data <- subset(data, dataset != "uwv")
data$method <- as.character(data$method)
data$dataset <- as.character(data$dataset)
#data$method[data$method=="single_threshold"] <- "Single Alarm"
data$dataset[data$dataset=="uwv_all"] <- "uwv"
data$dataset[data$dataset=="traffic_fines_1"] <- "traffic_fines"

data$ratio <- as.factor(data$ratio)
data$ratio_com <- as.factor(data$c_com)


data <- subset(data, cls=="rf")

head(data)

dt_multi <- subset(data, metric=="cost_avg" & method=="same_time_nonmyopic")
dt_alarm1 <- subset(data, metric=="cost_avg" & method=="opt_threshold")
dt_alarm2 <- subset(data, metric=="cost_avg" & method=="alarm2")
dt_single <- merge(dt_alarm1, dt_alarm2, by=c("dataset", "c_miss", "c_action", "c_postpone", "c_com", "early_type", "cls", "ratio"), suffixes=c("_alarm1", "_alarm2"))
dt_single$value_single <- dt_single$value_alarm1
dt_single$value_single <- ifelse(dt_single$value_single > dt_single$value_alarm2, dt_single$value_alarm2 ,dt_single$value_single)


dt_merged <- merge(dt_multi, dt_single, by=c("dataset", "c_miss", "c_action", "c_postpone", "c_com", "early_type", "cls", "ratio"), suffixes=c("_multi", "_single"))

print(subset(dt_merged,c_com > 19))

dt_merged$benefit <- dt_merged$value/dt_merged$value_single

min_value = 0.9
max_value = 1.1

dt_merged$benefit <- ifelse(dt_merged$benefit < min_value,min_value,dt_merged$benefit)
dt_merged$benefit <- ifelse(dt_merged$benefit > max_value,max_value,dt_merged$benefit)
dt_merged$benefit <-ifelse(((dt_merged$c_action * 1.2) + (dt_merged$c_com * 0.5)) < 0.9*(dt_merged$c_action + dt_merged$c_com),dt_merged$benefit,3000)



ggplot(subset(dt_merged, cls=="rf" & c_com > 0), aes(factor(ratio_com_alarm1), factor(ratio))) + geom_tile(aes(fill = benefit), colour = "black") + 
  theme_bw(base_size=base_size) + scale_fill_gradientn(colours=c("green4","green3","green2","white","red2","red3","red4"),breaks=c(0.95,1.0,1.05), limits=c(min_value,max_value),name="ratio") + 
  xlab("c_com") + ylab("c_out : c_in") +  theme(axis.text.x = element_text(size=20))  + facet_wrap( ~ dataset, ncol=3)
ggsave("/Users/stephanf/Dropbox/Dokumente/Masterstudium/Masterthesis/testresults/MultiAlarm/result_cost_avg_randomSplit_same_time_nonmyopic_vs_single.pdf", width = 20, height = 13)

