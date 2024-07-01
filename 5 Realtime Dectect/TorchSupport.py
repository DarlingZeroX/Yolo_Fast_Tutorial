import torch

def checkTorch():
    print("torch版本",torch.__version__)
    print("GPU是否可用",torch.cuda.is_available())
    print("GPU个数",torch.cuda.device_count())
    print("对应cudnn版本号",torch.backends.cudnn.version())
    print("对应cuda版本号",torch.version.cuda)
    print("--------------------------------------------------")
    #查看torchvision和torch版本是否匹配
    import torchvision
    print(torch.__version__)
    print(torchvision.__version__)

if __name__ == "__main__":
    checkTorch()