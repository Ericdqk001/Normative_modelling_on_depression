install.packages("poLCA")

library(MASS)
library(scatterplot3d)
library(poLCA)

# Load the data
cbcl_data <- read.csv("processed_data/cbcl_t_no_mis_dummy.csv")
