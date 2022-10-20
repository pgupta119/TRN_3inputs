# from .hdd_data_layer import TRNHDDDataLayer
# import lib.utils
# from .thumos_data_layer import TRNTHUMOSDataLayer
# from lib.datasets import hdd_data_layer
# from lib.datasets.hdd_data_layer import TRNHDDDataLayer
import sys
sys.path.append('/workspace/persistent/TRN.pytorch/lib/')
from hdd_data_layer import TRNHDDDataLayer
_DATA_LAYERS = {
    'TRNHDD': TRNHDDDataLayer,
    # 'TRNTHUMOS': TRNTHUMOSDataLayer,
}


def build_dataset(args, phase):
    data_layer = _DATA_LAYERS[args.model + args.dataset]
    return data_layer(args, phase)
