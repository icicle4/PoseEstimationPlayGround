import os
import torch
from torchsummary import summary


from model.pose.model_manager import ModelManager
from lib.tools.util.logger import Logger as Log
from lib.tools.parser.pose_parser import PoseParser
from lib.core.inference import get_outputs, aggregate_results
from lib.core.group import HeatmapParser
from tqdm import tqdm
import torchvision
import torch.backends.cudnn as cudnn

from lib.data import make_test_dataloader
from lib.utils.transforms import resize_align_multi_scale
from lib.utils.transforms import get_final_preds
from lib.utils.transforms import get_multi_scale_size


# markdown format output
def _print_name_value(logger, name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class HrPoseTest(object):
    def __init__(self, configer):
        self.configer = configer
        self.pose_parser = PoseParser(configer)
        self.pose_model_manager = ModelManager(configer)
        self.device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        self.pose_net = None
        self._init_model()

    def _init_model(self):
        self.model = self.pose_model_manager.get_multi_pose_model()
        summary(self.model, (3, self.configer.get("data", "input_size"), self.configer.get("data", "input_size")))
        if self.configer.get("test", "model_file"):
            Log.info("=> loading model from {}".format(self.configer.get("test", "model_file")))
        else:
            model_state_file = os.path.join(self.configer.get("final_output_dir"), "model_best.pth.tar")
            Log.info("=> loading model from {}".format(model_state_file))
            self.model.load_state_dict(torch.load(model_state_file))
        self.model = self.model.cuda()
        self.model.eval()

    def test(self):

        # cudnn related setting
        cudnn.benchmark = self.configer.get("cudnn", "benchmark")
        torch.backends.cudnn.deterministic = self.configer.get("cudnn", "determinstic")
        torch.backends.cudnn.enabled = self.configer.get("cudnn", "enabled")

        data_loader, test_dataset = make_test_dataloader(self.configer)
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

        parser = HeatmapParser(self.configer)
        all_preds = []
        all_scores = []

        pbar = tqdm(total=len(test_dataset)) if self.configer.get("test", "log_progress") else None

        for i, (images, annos) in enumerate(data_loader):
            assert 1 == images.size(0), 'Test batch size should be 1'

            image = images[0].cpu().numpy()
            # size at scale 1.0
            base_size, center, scale = get_multi_scale_size(
                image,
                self.configer.get("data", "input_size"),
                1.0,
                min(self.configer.get("scale_factor"))
            )

            with torch.no_grad():
                final_heatmaps = None
                tags_list = []

                for idx, s in enumerate(sorted(self.configer.get("test", "scale_factor"), reverse=True)):
                    input_size = self.configer.get("data", "input_size")
                    image_resized, center, scale = resize_align_multi_scale(
                        image, input_size, s, min(self.configer.get("test", "scale_factor"))
                    )
                    image_resized = transforms(image_resized)
                    image_resized = image_resized.unsqueeze(0).cuda()

                    outputs, heatmaps, tags = get_outputs(
                        self.configer, self.model, image_resized, self.configer.get("test", "flip_test"),
                        self.configer.get("test", "project2image"), base_size
                    )

                    final_heatmaps, tags_list = aggregate_results(
                        self.configer, s, final_heatmaps, tags_list, heatmaps, tags
                    )

                final_heatmaps = final_heatmaps / float(len(self.configer.get("test", "scale_factor")))
                tags = torch.cat(tags_list, dim=4)

                grouped, scores = parser.parse(
                    final_heatmaps, tags, self.configer.get("test", "adjust"),
                    self.configer.get("test", "refine")
                )
                final_results = get_final_preds(
                    grouped, center, scale,
                    [final_heatmaps.size(3), final_heatmaps.size(2)]
                )

                all_preds.append(final_results)
                all_scores.append(scores)

            if self.configer.get("test", "log_progress"):
                pbar.update()

            name_values, _ = test_dataset.evaluate(
                self.configer, all_preds, all_scores, self.configer.get("final_output_dir")
            )

            if isinstance(name_values, list):
                for name_value in name_values:
                    _print_name_value(Log, name_value, self.configer.get("model", "name"))
            else:
                _print_name_value(Log, name_values, self.configer.get("model", "name"))
