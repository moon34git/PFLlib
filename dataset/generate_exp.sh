# d - dataset
# nclasses - number of classes
# dir - directory
# niid - non-iid
# bal - balance
# par - partition (None or dir or pat) - dir not using cpc argument
# nclients - number of clients
# cpc - classes per client (for -par pat option)
# tr - train ratio
# a - alpha (for Dirichlet distribution)
# bs - batch size
# ls - least samples (guarantee that each client must have at least one samples for testing)
# spr - split ratio

#Generating codes -# Class 10
python generate_main.py -d mnist -nclasses 10 -dir mnist/mnist_test2 \
    -niid niid -bal unbalance -par dir -nclients 20 -cpc 2 -tr 0.8 -a 0.1 -bs 32 -ls 1 -spr 0.5 

python generate_main.py -d fmnist -nclasses 10 -dir fmnist/fmnist2 \
    -niid niid -bal unbalance -par pat -nclients 20 -cpc 2 -tr 0.8 -a 0.1 -bs 32 -ls 1 -spr 0.5 

python generate_main.py -d cifar10 -nclasses 10 -dir cifar10/cifar1 \
    -niid niid -bal unbalance -par pat -nclients 20 -cpc 2 -tr 0.8 -a 0.1 -bs 32 -ls 1 -spr 0.5

#Generating codes -# Class 100
python generate_main.py -d cifar100 -nclasses 100 -dir cifar100/cifar1 \
    -niid niid -bal unbalance -par dir -nclients 20 -cpc 20 -tr 0.8 -a 0.1 -bs 32 -ls 1 -spr 0.5

