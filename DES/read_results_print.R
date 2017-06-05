# setwd("~/GitProjects/CryptoGPU/DES")
library("ggplot2")

des_gpu_hubble_1 = read.table("./results_perf/des_gpu/hubble_1.log", header = FALSE, sep = "\t", stringsAsFactors = FALSE)
des_gpu_hubble_2 = read.table("./results_perf/des_gpu/hubble_2.log", header = FALSE, sep = "\t", stringsAsFactors = FALSE)
des_gpu_hubble_3 = read.table("./results_perf/des_gpu/hubble_3.log", header = FALSE, sep = "\t", stringsAsFactors = FALSE)
des_gpu_hubble_4 = read.table("./results_perf/des_gpu/hubble_4.log", header = FALSE, sep = "\t", stringsAsFactors = FALSE)

des_seq_hubble_1 = read.table("./results_perf/des_seq/hubble_1.log", header = FALSE, sep = "\t", stringsAsFactors = FALSE)
des_seq_hubble_2 = read.table("./results_perf/des_seq/hubble_2.log", header = FALSE, sep = "\t", stringsAsFactors = FALSE)
des_seq_hubble_3 = read.table("./results_perf/des_seq/hubble_3.log", header = FALSE, sep = "\t", stringsAsFactors = FALSE)
des_seq_hubble_4 = read.table("./results_perf/des_seq/hubble_4.log", header = FALSE, sep = "\t", stringsAsFactors = FALSE)



data_gpu = rbind(
  des_gpu_hubble_1,
  des_gpu_hubble_2,
  des_gpu_hubble_3,
  des_gpu_hubble_4
)

data_seq = rbind(
  des_seq_hubble_1,
  des_seq_hubble_2,
  des_seq_hubble_3,
  des_seq_hubble_4
)

#Leitura dos dados GPU
names(data_gpu) <- "output_line" 
results <- split(data_gpu, f = rep(1:(nrow(data_gpu)/10), each = 10))
print (results)
read_result <- function(df) {
  
  output <- lapply(strsplit(df$output_line, "[ :]"), function(x) x[x != ""])
  algorithm <- output[[1]][1]
  type <- output[[1]][2]
  filesize <- round(as.numeric(output[[2]][4])/1000000,2)
  time_allocation <- as.numeric(gsub(",", ".",output[[3]][4]))
  time_algorithm <- as.numeric(gsub(",", ".",output[[5]][4]))
  time_ioops <- as.numeric(gsub(",", ".",output[[7]][5]))
  time_elapsed <- as.numeric(gsub(",", ".",output[[9]][4]))
 
  results <- data.frame(
    algorithm = algorithm,
    type = type,
    filesize = filesize,
    time_allocation = time_allocation,
    time_algorithm = time_algorithm,
    time_ioops = time_ioops,
    time_elapsed = time_elapsed
  )
  
}

results_gpu <- do.call("rbind", lapply(results, read_result))

#Leitura dos dados SEQ
names(data_seq) <- "output_line" 
results <- split(data_seq, f = rep(1:(nrow(data_seq)/10), each = 10))

read_result <- function(df) {
  
  output <- lapply(strsplit(df$output_line, "[ :]"), function(x) x[x != ""])
  algorithm <- output[[1]][1]
  type <- output[[1]][2]
  filesize <- round(as.numeric(output[[2]][4])/1000000,2)
  time_allocation <- as.numeric(gsub(",", ".",output[[3]][4]))
  time_algorithm <- as.numeric(gsub(",", ".",output[[5]][4]))
  time_ioops <- as.numeric(gsub(",", ".",output[[7]][5]))
  time_elapsed <- as.numeric(gsub(",", ".",output[[9]][4]))
 
  results <- data.frame(
    algorithm = algorithm,
    type = type,
    filesize = filesize,
    time_allocation = time_allocation,
    time_algorithm = time_algorithm,
    time_ioops = time_ioops,
    time_elapsed = time_elapsed
  )
  
}

results_seq <- do.call("rbind", lapply(results, read_result))

print(results_seq)

fulldata = rbind(
  results_gpu,
  results_seq
)

write.csv(fulldata, file = "./results_perf/results.csv")

fig1 <- ggplot(data = fulldata, aes(x = as.factor(filesize), y = time_algorithm, color = type)) +
  geom_boxplot() +
  #   facet_wrap(~imsize, scales = "free", labeller = "label_both") +
  ggtitle("Tempo de Execução x Tamanho do Arquivo em MB processado") +
  xlab("Tamanho do Arquivo em MB processado") +
  ylab("Tempo de execução (s)") +
  guides(color = guide_legend(title="Implementação")) +
  theme(plot.title = element_text(hjust = 0.5, vjust = 5, size = 20),
        strip.text = element_text(size= 16),
        axis.title = element_text(size = 16),
        legend.title = element_text(size = 16),
        legend.text = element_text(size = 16),
        axis.text = element_text(size = 12),
        axis.title.y=element_text(vjust=5),
        axis.title.x=element_text(vjust=-10),
        plot.margin = unit(c(1,1,1,1), "cm"))

pdf("resultado_des.pdf", width = 16, height = 9)
print(fig1)
dev.off()



dados <- read.csv("./results_perf/results.csv")
library("dplyr")



select(dados, algorithm, type, filesize,time_elapsed,time_allocation,time_algorithm,time_ioops)

grouped_result <- group_by(dados, algorithm, type, filesize)

summarised_result <- summarise(grouped_result,
                               media_time_elapsed = mean(time_elapsed, na.rm = TRUE),
                               desv_pad_time_elapsed = sd(time_elapsed),
                               media_time_allocation = mean(time_allocation, na.rm = TRUE),
                               desv_pad_time_allocation = sd(time_allocation),
                               media_time_algorithm = mean(time_algorithm, na.rm = TRUE),
                               desv_pad_time_algorithm = sd(time_algorithm),
                               media_time_ioops = mean(time_ioops, na.rm = TRUE),
                               desv_pad_time_ioops = sd(time_ioops))


write.csv(summarised_result, file = "./results_perf/summarised_result.csv")  

fig2 <- ggplot(data=summarised_result, aes(x = factor(filesize), y=media_time_algorithm, fill=type)) +
  geom_bar(stat="identity", position=position_dodge())+
  #   facet_wrap(~imsize, scales = "free", labeller = "label_both") +
  ggtitle("Média do Tempo de Execução x Tamanho do Arquivo em MB processado") +
  xlab("Tamanho do Arquivo em MB processado") +
  ylab("Tempo de execução (s)") +
  guides(color = guide_legend(title="Implementação")) +
  theme(plot.title = element_text(hjust = 0.5, vjust = 5, size = 20),
        strip.text = element_text(size= 16),
        axis.title = element_text(size = 16),
        legend.title = element_text(size = 16),
        legend.text = element_text(size = 16),
        axis.text = element_text(size = 12),
        axis.title.y=element_text(vjust=5),
        axis.title.x=element_text(vjust=-10),
        plot.margin = unit(c(1,1,1,1), "cm"))

pdf("resultado_des_bar.pdf", width = 16, height = 9)
print(fig2)
dev.off()  


fig3 <- ggplot(data = summarised_result, aes(x = as.factor(filesize), y = media_time_algorithm, color = type,group = type)) +
  geom_errorbar(aes(ymin=media_time_algorithm-desv_pad_time_algorithm, ymax=media_time_algorithm+desv_pad_time_algorithm), width=.1) +
  geom_line() +
  geom_point()+
  ggtitle("Média do Tempo de Execução x Tamanho do Arquivo em MB processado",
          subtitle = "Média com Desvio Padrão") +
  xlab("Tamanho do Arquivo em MB processado") +
  ylab("Média do Tempo de execução (s)") +
  guides(color = guide_legend(title="Implementação")) +
  theme(plot.title = element_text(hjust = 0.5, vjust = 5, size = 20),
        strip.text = element_text(size= 16),
        axis.title = element_text(size = 16),
        legend.title = element_text(size = 16),
        legend.text = element_text(size = 16),
        axis.text = element_text(size = 12),
        axis.title.y=element_text(vjust=5),
        axis.title.x=element_text(vjust=-10),
        plot.margin = unit(c(1,1,1,1), "cm"))

pdf("resultado_des_media.pdf", width = 16, height = 9)
print(fig3)
dev.off()
  



























