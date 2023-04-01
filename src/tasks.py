from learned_optimization.tasks.fixed.image_mlp import ImageMLP_FashionMnist_Relu128x128
from learned_optimization.tasks.fixed.conv import Conv_Cifar10_32x64x64


def get_task(task):
    tasks = {
        "image_mlp" : ImageMLP_FashionMnist_Relu128x128(),
        "conv" : Conv_Cifar10_32x64x64(),
    }

    return tasks[task]