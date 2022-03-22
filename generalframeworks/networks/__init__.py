##### Get the Model #####
import torchvision.models as models
from .enet import Enet
from .enet_r import Enet_with_representation
from .deeplabv3.deeplabv3 import DeepLabv3Plus
def network_factory(name: str, num_class: int):
    net_dict = {'Enet': Enet(num_classes=num_class),
                'Enet_r': Enet_with_representation(num_classes=num_class),
                'DeepLabv3Plus': DeepLabv3Plus(models.resnet50(pretrained=True), num_classes=num_class)}
    print(name + ' has been prepared.')
    return net_dict[name]
