import torch 

def evaluate(model, dataloader, device):
    model.eval()
    model.to(device)

    outputs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            outputs.append({
                "logits": logits.cpu(),
                "labels": labels.cpu
            })
    return outputs


