import torch

def predictions(model, device, loader):
    model.eval()
    print(f"CUDA available: {torch.cuda.is_available()}")
    if (torch.cuda.is_available()):
        y_hat = torch.tensor([]).cuda()
        y_true = torch.tensor([]).cuda()
    else:
        y_hat = torch.tensor([]).to(device)
        y_true = torch.tensor([]).to(device)
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            if output.dim() == 0:
                output = output.unsqueeze(0)
            y_hat = torch.cat((y_hat, output),0) 
            y_true = torch.cat((y_true, data.y),0)
            
    return y_hat, y_true


def get_hardware_name():
    return 'cuda' if torch.cuda.is_available() else 'cpu'