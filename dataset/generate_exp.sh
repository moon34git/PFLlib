#Generating codes -# Class 10
#cpc is for -par pat option
#dir is not using cpc argument
#noniid must set par argument
#ls guarantee that each client must have at least one samples for testing. 
#alpha for Dirichlet distribution

python generate_main.py -d mnist -nclasses 10 -dir mnist/mnist_test2 -niid niid -bal unbalance -par dir -nclients 20 -cpc 2 -ts 0.8 -a 0.1 -bs 32 -ls 1
python generate_main.py -d femnist -nclasses 10 -dir femnist/femnist2 -niid niid -bal unbalance -par pat -nclients 20 -cpc 2 -ts 0.8 -a 0.1 -bs 32 -ls 1
python generate_main.py -d cifar10 -nclasses 10 -dir cifar10/cifar1 -niid niid -bal unbalance -par pat -nclients 20 -cpc 2 -ts 0.8 -a 0.1 -bs 32 -ls 1

#Generating codes -# Class 100
python generate_main.py -d cifar100 -nclasses 100 -dir cifar100/cifar1 -niid niid -bal unbalance -par dir -nclients 20 -cpc 20 -ts 0.8 -a 0.1 -bs 32 -ls 1