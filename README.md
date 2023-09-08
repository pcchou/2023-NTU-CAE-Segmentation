# NTU-CAE-Segmentation
My project on image segmentation for tile spalling detection at NTUCE CAE Division / NCREE Internship.

The aim is to provide a image segmentation mask for tile spalling of building exterior. We use U-Net, a Deep Learning-based model architecture for the job.

<img width="853" src="https://github.com/pcchou/NTU-CAE-Segmentation/assets/5615415/92627bf5-0ecd-4263-8bde-2249b36b409d">

## Files

- myunet.py: The code for the altered model architecture. It is modified from [segmentation-models-pytorch](https://smp.readthedocs.io/en/latest/_modules/segmentation_models_pytorch/decoders/unet/model.html#Unet).
- auto_evaluate.py: A script to automate the evaluation of trained models using testing data.
- Unet-efficientnet-b6-CrossEntropyLoss-4.pt: A sample of the model trained by us. You can check its prediction result by the following sample code:

```python
def predict_image_mask_miou(model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device); image=image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        output = model(image)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked
```

Some base code here has originated from [tanishqgautam/Drone-Image-Semantic-Segmentation](https://github.com/tanishqgautam/Drone-Image-Semantic-Segmentation).

## Poster

![新式影像分割模型於建物外牆破壞偵測之研發_周秉宇](https://github.com/pcchou/NTU-CAE-Segmentation/assets/5615415/0356c262-006a-4673-a854-20ba8b5cbf3c)
