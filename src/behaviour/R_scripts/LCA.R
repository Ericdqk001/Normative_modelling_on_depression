install.packages("reshape2")

library(MASS)
library(scatterplot3d)
library(poLCA)
library(dplyr)
library(reshape2)
library(ggplot2)
# Load the data and set subject id as the row names
cbcl_data <- read.csv("processed_data/cbcl_t_no_mis_dummy.csv", row.names = 1)

# Get the cbcl variable names
variable_names <- colnames(cbcl_data)

# Specify the formula for latent class analysis
f <- with(cbcl_data, cbind(cbcl_scr_syn_anxdep_t, cbcl_scr_syn_withdep_t,
                            cbcl_scr_syn_somatic_t, cbcl_scr_syn_social_t,
                            cbcl_scr_syn_thought_t, cbcl_scr_syn_attention_t,
                            cbcl_scr_syn_rulebreak_t, cbcl_scr_syn_aggressive_t)~1)

# Fit the model
lca_2 <- poLCA (f, cbcl_data, nclass=4, maxiter=50000, graphs=FALSE, nrep=10, verbose =TRUE)


#------ run a sequence of models with 2-5 classes and print out the model with the lowest BIC
# max_II <- -100000
# min_bic <- 100000
# for(i in 2:5){
#   lc <- poLCA(f, cbcl_data, nclass=i, maxiter=50000, na.rm=FALSE,
#               nrep=10, verbose=TRUE)
#   if(lc$bic < min_bic){
#     min_bic <- lc$bic
#     LCA_best_model<-lc
#   }
# }
# LCA_best_model

# LCA_best_model$cbcl_scr_syn_anxdep_t

# Class 4 results in a model with the lowest BIC (36349.93)

# Get the results of models with different class numbers into a dataframe
set.seed(01012)
lc1<-poLCA(f, cbcl_data, nclass=2, maxiter=50000, na.rm=FALSE,
              nrep=10, verbose=TRUE)
lc2<-poLCA(f, cbcl_data, nclass=3, maxiter=50000, na.rm=FALSE,
              nrep=10, verbose=TRUE)
lc3<-poLCA(f, cbcl_data, nclass=4, maxiter=50000, na.rm=FALSE,
              nrep=10, verbose=TRUE)
lc4<-poLCA(f, cbcl_data, nclass=5, maxiter=50000, na.rm=FALSE,
              nrep=10, verbose=TRUE)

# generate dataframe with fit-values

results <- data.frame(Modell=c("Classes"),
                      log_likelihood=lc1$llik,
                      df = lc1$resid.df,
                      BIC=lc1$bic)

results$Modell<-as.integer(results$Modell)
results[1,1]<-c("2 classes")
results[2,1]<-c("3 classes")
results[3,1]<-c("4 classes")
results[4,1]<-c("5 classes")

results[2,2]<-lc2$llik
results[3,2]<-lc3$llik
results[4,2]<-lc4$llik

results[2,3]<-lc2$resid.df
results[3,3]<-lc3$resid.df
results[4,3]<-lc4$resid.df

results[2,4]<-lc2$bic
results[3,4]<-lc3$bic
results[4,4]<-lc4$bic


# Save the predicted class and posterior probabilities to the cbcl_data dataframe
lc3$predclass

lc3$posterior

cbcl_data$predicted_class <- lc3$predclass

# Add the posterior probabilities as new columns in cbcl_data
# This assumes that lc3$posterior is a matrix with one column for each class
posterior_columns <- paste("ClassProb", 1:ncol(lc3$posterior), sep="_")
cbcl_data[posterior_columns] <- lc3$posterior

# Save the data with the predicted class and posterior probabilities
write.csv(cbcl_data, "processed_data/cbcl_t_no_mis_dummy_class_member.csv", row.names = TRUE)

# Visualise conditional probability of each variable for each class
lcmodel <- reshape2::melt(lc3$probs, level=2)
write.csv(lcmodel, "src/behaviour/files/lcmodel_prob_class.csv", row.names = TRUE)
