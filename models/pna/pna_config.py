import argparse

parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
parser.add_argument('--device', type=int, default=0,
                    help='which gpu to use if any (default: 0)')
parser.add_argument('--source', type=str, default="ogb",
                    help='which gpu to use if any (default: 0)')                    
parser.add_argument('--gnn', type=str, default='pna',
                    help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
parser.add_argument('--metrics', type=str, default="rocauc",
                    help="Evaluation metrics: rocauc, accuracy, mae")
parser.add_argument('--drop_ratio', type=float, default=0.5,
                    help='dropout ratio (default: 0.5)')
parser.add_argument('--num_layer', type=int, default=4,
                    help='number of GNN message passing layers (default: 5)')
parser.add_argument('--emb_dim', type=int, default=80,
                    help='dimensionality of hidden units in GNNs (default: 300)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training (default: 32)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--epochs', type=int, default=100,
                     help='number of epochs to train (default: 100)')
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of workers (default: 0)')
parser.add_argument('--dataset', type=str, default="ogbg-molbbbp",
                    help='dataset name (default: ogbg-molbbbp)')

parser.add_argument('--feature', type=str, default="full",
                    help='full feature or simple feature')
parser.add_argument('--filename', type=str, default="result.txt",
                    help='filename to output result (default: )')
parser.add_argument('--output_model_dir', type=str, default="save",
                    help='the path used for saving the best eval model')
parser.add_argument('--retrieval', type=int, default=0,
                    help='decide whether to use the retrieval results')
parser.add_argument('--k', type=int, default=0,
                    help='number of retrieval graphs')

args = parser.parse_args()
print("Args:")
print(args)
