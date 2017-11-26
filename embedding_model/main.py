import parser
import embedding
import train_helper
import sampler
import eval_metric
import argparse


def parse_args():
    """
    parse the embedding model arguments
    """
    parser_arg = argparse.ArgumentParser(description =
                                         "run embedding for name disambiguation")
    parser_arg.add_argument('data_source', help = 'data source')
    parser_arg.add_argument("file_path", help = 'input file name')
    parser_arg.add_argument("latent_dimen", type = int, default = 20,
                            help = 'number of dimension in embedding')
    parser_arg.add_argument("alpha", type = float, default = 0.02,
                            help = 'learning rate')
    parser_arg.add_argument("matrix_reg", type = float, default = 0.01,
                            help = 'matrix regularization parameter')
    parser_arg.add_argument("num_epoch", type = int, default = 100,
                            help = "number of epochs for SGD inference")
    parser_arg.add_argument("sampler_method", help = "sampling approach")
    return parser_arg.parse_args()


def main(args):
    """
    pipeline for representation learning for all papers for a given name reference
    """
    dataset = parser.DataSet(args.file_path)
     
    if args.data_source == "arnetminer":
        dataset.reader_arnetminer()
    elif args.data_source == "citeseerx":
        dataset.reader_citeseerx()

    bpr_optimizer = embedding.BprOptimizer(args.latent_dimen, args.alpha,
                                           args.matrix_reg)
    pp_sampler = sampler.CoauthorGraphSampler()
    pd_sampler = sampler.BipartiteGraphSampler()
    dd_sampler = sampler.LinkedDocGraphSampler()
    eval_f1 = eval_metric.Evaluator()

    run_helper = train_helper.TrainHelper()
    run_helper.helper(args.num_epoch, dataset, bpr_optimizer,
                      pp_sampler, pd_sampler, dd_sampler,
                      eval_f1, args.sampler_method)


if __name__ == "__main__":
    args = parse_args()
    main(args)
