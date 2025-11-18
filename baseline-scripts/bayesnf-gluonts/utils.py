import torch


def check_gpu_status():
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    # print(torch.cuda.current_device())
    # print(torch.cuda.device(0))
    # print(torch.cuda.get_device_name(0))