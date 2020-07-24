import argparse


def get_params():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]
    parser = argparse.ArgumentParser(description="Recognition params")

    # VAD
    parser.add_argument('--use_vad', type=_str_to_bool, default=True)
    parser.add_argument('--vad_mode', type=int, default=3)
    parser.add_argument('--vad_frame_dur', type=int, default=30)
    parser.add_argument('--vad_buffer_length', type=int, default=10)

    # preprocess
    parser.add_argument('--train_data_dir', type=str, default='data/train')
    parser.add_argument('--train_data_txt_dir', type=str, default='data/train.txt')
    parser.add_argument('--mel_output_dir', type=str, default='data/mel')
    parser.add_argument('--train_proc_dir', type=str, default='data/train_proc')
    parser.add_argument('--n_mels', type=int, default=24)
    parser.add_argument('--mel_frame_length', type=int, default=400)
    # 按照原文说法 这里帧长应该是400 即25ms
    parser.add_argument('--mel_frame_shift', type=int, default=160)

    parser.add_argument("--window_size", type=int, default=300)
    # 当前帧移为10ms，后续需要调整
    parser.add_argument('--feature_dim', type=int, default=24)
    parser.add_argument('--lan_num', type=int, default=17)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_save_path', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    # train_step = train_dataset_len / batch_size = 10556 * 0.9 / 32
    parser.add_argument('--train_step', type=int, default=300)
    parser.add_argument('--validate_step', type=int, default=35)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--check_point_distance', type=int, default=5)
    parser.add_argument('--model', type=str, default='xvecTDNN')
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument('--load_epoch', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--eps', type=float, default=1e-5)

    parser.add_argument('--predict_step', type=int, default=1)
    parser.add_argument('--predict_input', type=str, default='sample/')
    parser.add_argument('--predict_output', type=str, default='data/predict.csv')

    return parser.parse_args()

