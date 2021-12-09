def add_args(parser):
    parser.add_argument(
        '--data_path', type=str, default='../datasets/wikitext103_raw_gpt2bpe.pkl'
    )
    parser.add_argument(
        '--chunk_size_train', type=int, default=1024
    )
    parser.add_argument(
        '--chunk_size_valid', type=int, default=1024
    )
    parser.add_argument(
        '--token_limit_train', type=int, default=1024
    )
    parser.add_argument(
        '--token_limit_valid', type=int, default=1024
    )
    parser.add_argument(
        '--context_length', type=int, default=10
    )
    parser.add_argument(
        '--train_model_path', type=str, default=None
    )
    parser.add_argument(
        '--test_model_path', type=str, default=None
    )
    parser.add_argument(
        '--score_mle_model_path', type=str, default=None
    )
    parser.add_argument(
        '--tokenizer_cache_path', type=str, default='../models/tokenizer_cache/'
    )
    parser.add_argument(
        '--transformers_cache_path', type=str, default='../models/transformers_cache/'
    )
    parser.add_argument(
        '--max_epochs', type=int, default=100
    )
    parser.add_argument(
        '--lr', type=float, default=6.25e-5
    )
    parser.add_argument(
        '--adam_epsilon', type=float, default=1e-8
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0.01
    )
    parser.add_argument(
        '--warmup_steps', type=int, default=0
    )
    parser.add_argument(
        '--max_grad_norm', type=float, default=1.0
    )
    parser.add_argument(
        '--deterministic_decoding', type=str, default='greedy'
    )
    parser.add_argument(
        '--eval_context_length', type=int, default=10
    )
    parser.add_argument(
        '--decoding_max_length', type=int, default=500
    )
    parser.add_argument(
        '--decoding_len_factor', type=float, default=1.3
    )
    parser.add_argument(
        '--fixed_length', type=int, default=-1
    )
    parser.add_argument(
        '--print_every', type=int, default=100
    )
    parser.add_argument(
        '--patience', type=int, default=10
    )
    parser.add_argument(
        '--no_checkpoint', action='store_true'
    )
    parser.add_argument(
        '--seed', type=int, default=42
    )
    args = parser.parse_args()
    return args
