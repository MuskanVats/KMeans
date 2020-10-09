#Importing the data
data=read.csv("Mall_Customers.csv")


data=data[-1,]
X=data[,4:5]

dendrogram=hclust(dist(X,method="euclidean"), method='ward.D')
plot(dendrogram, main=paste("Optimum Number of Cluster of Clients"),
     xlab="Data Points", ylab="Euclidean Distances")

# Fitting Clustering to the dataset
hc=hclust(dist(X,method="euclidean"), method='ward.D')
y_hc=cutree(hc,5)


#Visualizing the clusters
library(cluster)
clusplot(X,y_hc, lines=0, shade=TRUE, color=TRUE, labels=2,
         plotchar=FALSE, span=TRUE, main=paste("Cluster of Clients"),
         xlab="Annual Income", ylab="Spending Score")