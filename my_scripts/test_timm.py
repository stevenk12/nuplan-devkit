from urllib.request import urlopen
from PIL import Image
import timm

# img = Image.open(urlopen(
#     'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
# ))

model = timm.create_model(
    'resnet50.a1_in1k',
    pretrained=True,
    features_only=True,
)
print(model)
# model = model.eval()

# # get model specific transforms (normalization, resize)
# data_config = timm.data.resolve_model_data_config(model)
# transforms = timm.data.create_transform(**data_config, is_training=False)

# output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

# for o in output:
#     # print shape of each feature map in output
#     # e.g.:
#     #  torch.Size([1, 64, 112, 112])
#     #  torch.Size([1, 256, 56, 56])
#     #  torch.Size([1, 512, 28, 28])
#     #  torch.Size([1, 1024, 14, 14])
#     #  torch.Size([1, 2048, 7, 7])

#     print(o.shape)
