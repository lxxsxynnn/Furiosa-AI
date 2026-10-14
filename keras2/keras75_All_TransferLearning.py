import warnings
warnings.filterwarnings('ignore')   # 콘솔창 경고문구 안보이게 하는 옵션

from keras.applications import VGG16, VGG19
from keras.applications import ResNet50, ResNet50V2
from keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from keras.applications import DenseNet201, DenseNet169, DenseNet121
from keras.applications import InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2
from keras.applications import MobileNetV3Small, MobileNetV3Large
from keras.applications import NASNetMobile, NASNetLarge
from keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
from keras.applications import Xception

model_list = [VGG16(), VGG19(), ResNet50(), ResNet50V2(), ResNet101(), ResNet101V2(), ResNet152(), ResNet152V2(),
              DenseNet201(), DenseNet169(), DenseNet121(), InceptionV3(), InceptionResNetV2(),
              MobileNet(), MobileNetV2(), MobileNetV3Small(), MobileNetV3Large(), NASNetMobile(), NASNetLarge(),
              EfficientNetB0(), EfficientNetB1(), EfficientNetB2(),
              ]

model = Xception()  # Xception의 summary와 가중치 먼저 확인

model.trainable = True
model.summary()

'''
Model: "xception"
______________________________________________________________________________________________________
 Layer (type)                        Output Shape          Param #     Connected to                     
======================================================================================================
 input_23 (InputLayer)              [(None, 299, 299, 3)]  0           []                               
 block1_conv1 (Conv2D)              (None, 149, 149, 32    864         ['input_23[0][0]']
 ...
 block14_sepconv2_act (Activation)  (None, 10, 10, 2048)   0           ['block14_sepconv2_bn[0][0]']                                                                  
 avg_pool (GlobalAveragePooling2D)  (None, 2048)           0           ['block14_sepconv2_act[0][0]']                                          
 predictions (Dense)                (None, 1000)           2049000     ['avg_pool[0][0]']               
======================================================================================================
Total params: 22,910,480
Trainable params: 22,855,952
Non-trainable params: 54,528
'''

print("모델명: ", model.name)   # 모델명: xception
print("전체 가중치 개수: ", len(model.weights))     # 전체 가중치 개수:  236
print("전체 훈련 가능 가중치 개수: ", len(model.trainable_weights))     # 전체 훈련 가능 가중치 개수:  156

'''
모델명             전체 가중치  훈련 가능 가중치
xception                 236   156
vgg16                     32    32
vgg19                     38    38
resnet50                 320   214
resnet50v2               272   174
resnet101                626   418
resnet101v2              544   344
resnet152                932   622
resnet152v2              816   514
densenet201             1006   604
densenet169              846   508
densenet121              606   364
inception_v3             378   190
inception_resnet_v2      898   490
mobilenet_1.00_224       137    83
mobilenetv2_1.00_224     262   158
MobileNetV3Small         210   142
MobileNetV3Large         266   174
nasnet_mobile           1126   742
nasnet_large            1546  1018
efficientnetb0           314   213
efficientnetb1           442   301
efficientnetb2           442   301

[VGG16, VGG19]
전체 == 훈련 가능
(Conv2D/Dense 의 kernel, bias 만 있음 : VGG16 16층 x 2 = 32)

[나머지]
전체 > 훈련 가능
 -> model.trainable = True 로 둔 Xception 도 236 / 156 으로 다름
 -> BatchNormalization의 moving_mean·moving_variance는 역전파로 안 바뀌는 가중치라 훈련 가능에서 빠짐 (VGG에는 BatchNormalization이 없음)
 '''