import matplotlib.pyplot as plt
from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import load_coco_data, decode_captions
from core.bleu import evaluate


def main(params):
  batch_size = params['batch_size']
  n_epochs = params['epoch']
  n_time_step = params['n_time_step']
  learning_rate = params['lr']
  model_path = params['model_path']
  log_path = params['log_path']

  data = load_coco_data(data_path='./data', split='train')
  word_to_idx = data['word_to_idx']
  val_data = load_coco_data(data_path='./data', split='val')

  model = CaptionGenerator(word_to_idx, dim_feature=[196, 512], dim_embed=256,
                                       dim_hidden=1024, n_time_step=n_time_step, prev2out=True, 
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

  solver = CaptioningSolver(model, data, val_data, n_epochs=n_epochs, batch_size=batch_size, update_rule='adam',
                                          learning_rate=learning_rate, print_every=3000, save_every=1, image_path='./image/',
                                    pretrained_model=None, model_path=model_path, test_model='./model/lstm/model-10',
                                     print_bleu=True, log_path=log_path)
  
  solver.train()

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  parser.add_argument('--batch_size', required=True, type=int, help='mini batch size when training the model')
  parser.add_argument('--epoch', required=True, type=int, help='the number of epoch for training the model')
  parser.add_argument('--n_time_step', default=16, type=int, help='time step size of lstm; this should be equal to max length of a caption + 1 in prepro.py')
  parser.add_argument('--lr', default=0.001, type=int, help='learning rate')
  parser.add_argument('--model_path', default='./model/lstm', help='path for saving checkpoint file')
  parser.add_argument('--log_path', default='./log/', help='log path for tensorboard visualization')

  args = parser.parse_args()
  params = vars(args) 
  main(params)