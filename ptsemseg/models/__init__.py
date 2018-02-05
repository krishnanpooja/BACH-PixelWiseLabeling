import torchvision.models as models

from ptsemseg.models.fcn import fcn8s


def get_model(name, n_classes):
    model = fcn8s(n_classes)
    #model = unet(n_classes)
    #model  = model(n_classes=n_classes)
    vgg16  = models.vgg16(pretrained=True)
    model.init_vgg16_params(vgg16)
  
    return model

def _get_model_instance(name):
    try:
        return {
            'fcn8s': fcn8s
        }[name]
    except:
        print('Model {} not available'.format(name))
