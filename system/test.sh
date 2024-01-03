#Training
python main.py -data mnist -nb 10 -m cnn -lbs 32 -lr 0.001 -ldg 0.99 -gr 3 -ls 1 -algo FedAvg -jr 1 -nc 20 -dir ../dataset/mnist/mnist2

#Arguments
-data datasetname
-nb number of class
-m networks cnn, dnn, resnet
-lbs local batch size
-lr learning rate
-ldg learning rate decay
-gr grobal round
-ls local epoch
-algo FedAvg, FedProx, FedScaffold, ...
-jr join round
-nc number of clients
-dir dataset directory
```

