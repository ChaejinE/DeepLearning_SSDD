from tf_2.segmentation.ssdd.train.trainer import SegTrainer, DecTrainer, SDTrainer, SUTrainer

import yaml
import argparse


class Train:
    def __init__(self, configure):
        self.cfg = configure
        self.trainer = self._init_trainer()

    def _init_trainer(self):
        if self.cfg['train_network'] == 'seg':

            return SegTrainer(self.cfg)
        elif self.cfg['train_network'] == 'dec':

            return DecTrainer(self.cfg)
        elif self.cfg['train_network'] == 'sd':

            return SDTrainer(self.cfg)
        elif self.cfg['train_network'] == 'su':

            return SUTrainer(self.cfg)
        else:
            raise ValueError('Unknown train model : {}'.format(self.cfg['train_model']))

    def train(self):
        self.trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', default='./train.yml')
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    Train(config).train()
