import argparse


def _add_common_args(arg_parser):
    arg_parser.add_argument('--config', type=str)

    # Input
    arg_parser.add_argument('--types_path', type=str, help="Path to type specifications")

    # Preprocessing
    arg_parser.add_argument('--tokenizer_path', type=str, help="Path to tokenizer")
    arg_parser.add_argument('--max_span_size', type=int, default=10, help="Maximum size of spans")
    arg_parser.add_argument('--lowercase', action='store_true', default=False,
                            help="If true, input is lowercased during preprocessing")
    arg_parser.add_argument('--sampling_processes', type=int, default=4,
                            help="Number of sampling processes. 0 = no multiprocessing for sampling")
    arg_parser.add_argument('--sampling_limit', type=int, default=100, help="Maximum number of sample batches in queue")

    # Logging
    arg_parser.add_argument('--label', type=str, help="Label of run. Used as the directory name of logs/models")
    arg_parser.add_argument('--log_path', type=str, help="Path do directory where training/evaluation logs are stored")
    arg_parser.add_argument('--store_predictions', action='store_true', default=False,
                            help="If true, store predictions on disc (in log directory)")
    arg_parser.add_argument('--store_examples', action='store_true', default=False,
                            help="If true, store evaluation examples on disc (in log directory)")
    arg_parser.add_argument('--example_count', type=int, default=None,
                            help="Count of evaluation example to store (if store_examples == True)")
    arg_parser.add_argument('--debug', action='store_true', default=False, help="Debugging mode on/off")

    # Model / Training / Evaluation
    arg_parser.add_argument('--model_path', type=str, help="Path to directory that contains model checkpoints")
    arg_parser.add_argument('--model_type', type=str, default="spert", help="Type of model")
    arg_parser.add_argument('--bert_type', type=str, default='bert', help='Type of bert')
    arg_parser.add_argument('--cpu', action='store_true', default=False,
                            help="If true, train/evaluate on CPU even if a CUDA device is available")
    arg_parser.add_argument('--eval_batch_size', type=int, default=1, help="Evaluation batch size")
    arg_parser.add_argument('--max_pairs', type=int, default=1000,
                            help="Maximum entity pairs to process during training/evaluation")
    arg_parser.add_argument('--boundary_filter_threshold', type=float, default=0.4,
                            help="Filter threshold for boundaries")
    arg_parser.add_argument('--size_embedding', type=int, default=25, help="Dimensionality of size embedding")
    arg_parser.add_argument('--prop_drop', type=float, default=0.1, help="Probability of dropout used in SpERT")
    arg_parser.add_argument('--freeze_transformer', action='store_true', default=False, help="Freeze BERT weights")
    arg_parser.add_argument('--no_overlapping', action='store_true', default=False,
                            help="If true, do not evaluate on overlapping entities "
                                 "and relations with overlapping entities")
    arg_parser.add_argument('--boundary_represent_mode', type=int, default=2,
                            help="Number of token of boundary representation")
    arg_parser.add_argument('--BD_include_type', action='store_true', default=False,
                            help="weather include type when boundary detecting")
    arg_parser.add_argument('--use_control_attention', action='store_true', default=False,
                            help="use_control_attention")

    arg_parser.add_argument('--use_sent_ctx', action='store_true', default=True,
                            help="use context of sentence")

    arg_parser.add_argument('--use_size_embedding', action='store_true', default=True,
                            help="_use_size_embedding")
    arg_parser.add_argument('--detect_upper', action='store_true', default=False,
                            help="detect upper char for every token")
    arg_parser.add_argument('--detect_entity_token', action='store_true', default=False,
                            help='detect entity token')
    arg_parser.add_argument('--detect_boundary', action='store_true', default=True,
                            help='detect detect_boundary')

    # Misc
    arg_parser.add_argument('--seed', type=int, default=None, help="Seed")
    arg_parser.add_argument('--cache_path', type=str, default=None,
                            help="Path to cache transformer models (for HuggingFace transformers library)")


def train_argparser():
    arg_parser = argparse.ArgumentParser()

    # Input
    arg_parser.add_argument('--train_path', type=str, help="Path to train dataset")
    arg_parser.add_argument('--valid_path', type=str, help="Path to validation dataset")

    # Logging
    arg_parser.add_argument('--save_path', type=str, help="Path to directory where model checkpoints are stored")
    arg_parser.add_argument('--init_eval', action='store_true', default=False,
                            help="If true, evaluate validation set before training")
    arg_parser.add_argument('--save_optimizer', action='store_true', default=False,
                            help="Save optimizer alongside model")
    arg_parser.add_argument('--train_log_iter', type=int, default=1, help="Log training process every x iterations")
    arg_parser.add_argument('--final_eval', action='store_true', default=False,
                            help="Evaluate the model only after training, not at every epoch")

    # Model / Training
    arg_parser.add_argument('--train_batch_size', type=int, default=2, help="Training batch size")
    arg_parser.add_argument('--epochs', type=int, default=20, help="Number of epochs")
    arg_parser.add_argument('--neg_entity_count', type=int, default=100,
                            help="Number of negative entity samples per document (sentence)")
    arg_parser.add_argument('--neg_boundary_count', type=int, default=100,
                            help="Number of negative boundary samples per document (sentence)")

    arg_parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
    arg_parser.add_argument('--lr_warmup', type=float, default=0.1,
                            help="Proportion of total train iterations to warmup in linear increase/decrease schedule")
    arg_parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay to apply")
    arg_parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm")

    _add_common_args(arg_parser)

    return arg_parser


def eval_argparser():
    arg_parser = argparse.ArgumentParser()

    # Input
    arg_parser.add_argument('--dataset_path', type=str, help="Path to dataset")

    _add_common_args(arg_parser)

    return arg_parser
