import argparse
from generators.generators import generate_mnist, generate_fmnist, generate_cifar10, generate_cifar100

def run(args):
    if args.data_dist == 'niid':
        niid = True
    else:
        niid = False
    if args.balance == 'balance':
        balance = True
    else:
        balance = False
    if args.dataset == 'mnist':
        generate_mnist(args.dir_path, args.num_clients, args.num_classes, niid, balance, args.partition, args.class_per_client, args.train_size,
                       args.alpha, args.batch_size, args.least_samples)
    elif args.dataset == 'fmnist':
        generate_fmnist(args.dir_path, args.num_clients, args.num_classes, niid, balance, args.partition, args.class_per_client, args.train_size,
                       args.alpha, args.batch_size, args.least_samples)
    elif args.dataset == 'cifar10':
        generate_cifar10(args.dir_path, args.num_clients, args.num_classes, niid, balance, args.partition, args.class_per_client, args.train_size,
                       args.alpha, args.batch_size, args.least_samples)
    elif args.dataset == 'cifar100':
        generate_cifar100(args.dir_path, args.num_clients, args.num_classes, niid, balance, args.partition, args.class_per_client, args.train_size,
                       args.alpha, args.batch_size, args.least_samples)
    else:
        print("Dataset not found!")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dataset", type=str, default="mnist", 
                        help="The name of dataset", choices=['mnist', 'fmnist', 'cifar10', 'cifar100'])
    parser.add_argument('-nclasses', "--num_classes", type=int, default=10,
                        help="The number of classes")
    parser.add_argument('-dir', '--dir_path', type=str, default="dataset/",
                        help="The path of dataset")
    parser.add_argument('-niid', '--data_dist', type=str, default='niid',
                        help="The data distribution is non-iid or not", choices=['iid', 'niid'])
    parser.add_argument('-bal', "--balance", type=str, default='balance',
                        help="The data distribution is balanced or not", choices=['balance', 'unbalance'])
    parser.add_argument('-par', "--partition", type=str, default=None,
                        help="The partition of data")
    parser.add_argument('-nclients', "--num_clients", type=int, default=20,
                        help="The number of clients")
    parser.add_argument('-cpc', "--class_per_client", type=int, default=2)
    parser.add_argument('-ts', "--train_size", type=float, default=0.75)
    parser.add_argument('-a', "--alpha", type=float, default=0.1)
    parser.add_argument('-bs', "--batch_size", type=int, default=32)
    parser.add_argument('-ls', "--least_samples", type=int, default=1)
    args = parser.parse_args()
    run(args)


