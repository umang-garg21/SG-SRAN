import argparse


class Argparser:
    """
    The actual argparser
    """

    def __init__(self):
        self.args = self.prepare_arg_parser().parse_args()

    def prepare_arg_parser(self):
        """
        Add all args to the argparser
        """

        arg_parser = argparse.ArgumentParser()

        # Hardware specifications

        arg_parser.add_argument(
            "--model_name",
            type=str,
            default="san_ortho_sec_data_L1_new",
            help="Name of Model you want to generate IPF map",
        )
        arg_parser.add_argument(
            "--dataset_type", type=str, default="Val", help="Val or Test Dataset"
        )
        arg_parser.add_argument(
            "--data", type=str, default="Ti64", help="type of material"
        )

        arg_parser.add_argument(
            "--model_to_load",
            type=str,
            default="model_best",
            help="which model to load",
        )
        arg_parser.add_argument(
            "--file_type", type=str, default="sr", help="[sr, hr, lr]"
        )
        arg_parser.add_argument(
            "--section", type=str, default="X_Block", help="[X_Block, Y_Block, Z_Block]"
        )
        arg_parser.add_argument(
            "--exp_type",
            type=str,
            default="min_angle_transform",
            help="type of experiment",
        )
        arg_parser.add_argument(
            "--exp_dir_path",
            type=str,
            default="/data/home/umang/Materials/QRBSA-jan30-joaquin-edit/Quaternion_experiments/",
            help="Path to the experiment directory",
        )
        arg_parser.add_argument(
            "--material_dream3dfile",
            type=str,
            default="/data/home/umang/Materials/Materials_data_mount/Open_718_Training.dream3d",
            help="Path to the material dream3d file",
        )

        return arg_parser
