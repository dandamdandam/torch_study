import torch
import torchvision.models as models

# 모델을 로컬로 저장하고 불러오기

#model = models.vgg16(weights='IMAGENET1K_V1')
#torch.save(model.state_dict(), 'model_weights.pth')
#
#model = models.vgg16() 
#model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
#model.eval()
#
#torch.save(model, 'model.pth')

"""
Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /home/dandamdandam/.cache/torch/hub/checkpoints/vgg16-397923af.pth
100.0%
"""

model = torch.load('model.pth', weights_only=False)

# 불러온 모델 활용하기
output = model(torch.randn(1, 3, 224, 224))
print(output)

"""
tensor([[-1.1079e-02,  4.2089e+00, -1.4231e+00, -1.4550e+00,  5.5710e-01,
          3.5526e+00,  3.6382e+00,  1.4296e+00,  1.1436e+00, -1.0994e+00,
          1.7337e-01,  2.4568e+00,  1.1814e+00,  9.1745e-01,  7.2926e-01,
          1.8408e+00, -2.2178e+00, -1.4921e+00,  1.5401e+00, -1.0821e+00,
생략..
"""