#Training
python main.py -data mnist -nb 10 -m cnn -lbs 32 -lr 0.001 -ldg 0.99 -gr 3 -ls 1 -algo FedAvg -jr 1 -nc 20 -dir ../dataset/mnist/mnist2