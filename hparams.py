import argparse


def get_params():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]
    parser = argparse.ArgumentParser(description="Recognition params")

    parser.add_argument("--window_size", type=int, default=200)
    # 当前帧移为10ms，总共的窗长为500ms，后续需要调整
    parser.add_argument('--feature_dim', type=int, default=24)
    parser.add_argument('--lan_num', type=int, default=17)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_save_path', type=str)
    parser.add_argument('--lr', type=float, default=1e-3)
    # train_step = train_dataset_len / batch_size = 10556 * 0.9 / 32
    parser.add_argument('--train_step', type=int, default=300)
    parser.add_argument('--validate_step', type=int, default=35)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--check_point_distance', type=int, default=1)
    parser.add_argument('--model', type=str, default='xvecTDNN')
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--train_txt_dir', type=str, default='data/train.txt')
    parser.add_argument('--load_epoch', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--eps', type=float, default=1e-5)

    return parser.parse_args()

