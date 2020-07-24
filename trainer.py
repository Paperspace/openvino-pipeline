import click
import logging
import os
import requests
import urllib.request
import zipfile
import matplotlib.pyplot as plt
from fastai.vision import *
from fastai.metrics import error_rate, mean_squared_error, top_k_accuracy
#from fastai.callbacks.tensorboard import *
from tbcallback import ImageGenTensorboardWriter
from tensorboardX import SummaryWriter
from bs4 import BeautifulSoup
import hydra
from omegaconf import DictConfig
from typing import Any
from torch.autograd import Variable
import torch
log = logging.getLogger(__name__)

def pull_data(dataset: DictConfig):
    if not os.path.exists(dataset.raw):
        os.makedirs(dataset.raw)
    if not os.path.exists(dataset.file):
        os.makedirs(dataset.file)

    url = dataset.url
    response= requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    links = [tag['href'] for tag in soup.findAll('a')]

    for link in links:
        if ".zip" in link:
            file_name = link.partition("/dibas/")[2]
            urllib.request.urlretrieve(link, dataset.raw + file_name) 
            zip_ref = zipfile.ZipFile(dataset.raw + file_name, 'r')
            zip_ref.extractall(dataset.file)   
            zip_ref.close()
            log.info("Downloaded and extracted: " + file_name)

    log.info(len(os.listdir((dataset.file))))

def validate_data(dataset: DictConfig):
    verify_images(dataset.file, delete=True, max_size=500, max_workers=1)

def train_model(cfg: DictConfig):
    bs = 64
    fnames = get_image_files(cfg.dataset.file)
    pat = r'/([^/]+)_\d+.tif$'
    data = ImageDataBunch.from_name_re(cfg.dataset.file, fnames, pat, ds_tfms=get_transforms(), size=24, bs=bs).normalize(imagenet_stats)
    print(data)
    #print(data.shape)
    #data.show_batch(rows=3, figsize=(7,8))
    learn = create_cnn(data, models.resnet34, metrics=accuracy)
    #learn = create_cnn(data, models.resnet50, metrics=error_rate)
    #project_id = 'projct1'
    ##tboard_path = Path('data/tensorboard/' + project_id)

    writer = SummaryWriter(comment='Demo', log_dir='/artifacts')

    # Track_weight and track_grad are used to decide if weights and gradients will be logged in TensorBoard
    # Metric names are names to be displayed in Tensorboard. The first is always validation loss
    # The order of metric names has to be the same than in learn.metrics
    #mycallback = partial(TensorBoardFastAI, writer, track_weight=True, track_grad=True, show_images=True, metric_names=['val loss', 'accuracy'])
    #learn.callback_fns.append(mycallback)
    
    learn.lr_find()
    #learn.recorder.plot()
    learn.callback_fns.append(partial(ImageGenTensorboardWriter, base_dir='/artifacts/', name='run1'))

    learn.fit_one_cycle(4)


    #learn.fit_one_cycle(8)

    path = learn.save('stage-1-50', True)
    learn.unfreeze()
    learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-4))

    #learn.load('stage-1-50');
# interp = ClassificationInterpretation.from_learner(learn)
# interp.plot_confusion_matrix(data.classes)

    preds,y,losses = learn.get_preds(with_loss=True)
    interp = ClassificationInterpretation(learn, preds, y, losses)
    interp.plot_top_losses(9, figsize=(7,7))

    print(preds)
    print(y)
    print(losses)
    errors = error_rate(preds, y)
    log.info("MSE: "+str(errors.double()))
    #top_k_accuracy = top_k_accuracy(preds, y, 1)
    #log.info("Accuracy: "+top_k_accuracy)

    import json

    gradient_metadata = {}
    gradient_metadata['metrics'] = []
    gradient_metadata['metrics'].append({
        'MSE': errors.data.tolist(),
        #'Accuracy': top_k_accuracy.double(),
        #'from': 'Nebraska'
    })

    with open('/artifacts/gradient-model-metadata.json', 'w') as outfile:
        json.dump(gradient_metadata, outfile)
        
    #interp.plot_confusion_matrix(plot_txt=False)
    path = learn.save('model', True)

    from torch.autograd import Variable
    sz = 24
    model = torch.load(path)
    model_name = "/artifacts/bacteria_classifier.onnx"

    dummy_input = Variable(torch.randn(1, 3, sz, sz)) # for iOS app, we predict only 1 image at a time, we don't use batch
    torch.onnx.export(learn.model, dummy_input, model_name, \
                    input_names=['image'], output_names=['bacterialclassifier'], verbose=True)


    return path


def export_to_open_vino(cfg: DictConfig):
    model_name = "/artifacts/bacteria_classifier.onnx"
    os.system("python /model-optimizer/mo.py --input_model /artifacts/bacteria_classifier.onnx --output_dir /artifacts/1/")


@hydra.main(config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    log.info(cfg.pretty())
    #pull_data(cfg.dataset)
    #validate_data(cfg.dataset)
    train_model(cfg)
    export_to_open_vino(cfg)
    #deploy_as_endpoint(cfg)
    #make_queries(cfg)

if __name__ == "__main__":
    my_app()