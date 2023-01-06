###########################################
# Trophic network for GUD implementations #
#                                         #
# Maps F. 2017                            #
###########################################


data <- read.csv("palatability.csv")

require(qgraph)

# Circle
qc <- qgraph(as.matrix(data), layout="circle")

# Hubs
qh <- qgraph(as.matrix(data))

# All the info you need is inside the returned object, including the layout of the graph.
# See qgraph help.