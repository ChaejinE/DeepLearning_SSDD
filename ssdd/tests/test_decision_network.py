import unittest
import numpy as np
from tf_2.segmentation.ssdd.model.decision_network import DecisionNet


class TestDecisionNetwork(unittest.TestCase):

    official_input_size = [(176, 64), (88, 32)]
    batch_size = 3
    num_class = 2

    def test_network(self):
        # TEST 1: 모델의 네트워크가 제대로 만들어 지는지?
        height, width = self.official_input_size[0]
        input_1 = np.random.random_sample(size=(self.batch_size, height, width, 1))
        input_2 = np.random.random_sample(size=(self.batch_size, height, width, 1024))
        decision_model = DecisionNet().network(mask_shape=input_1.shape,
                                               feature_shape=input_2.shape,
                                               num_class=self.num_class)
        output = decision_model([input_1, input_2])
        self.assertEqual((self.batch_size, self.num_class), np.shape(output))

        # TEST 2: official input size 중 절반 크기로 줄인 Input image size 도 통과 되는지?
        height, width = self.official_input_size[1]
        input_1 = np.random.random_sample(size=(self.batch_size, height, width, 1))
        input_2 = np.random.random_sample(size=(self.batch_size, height, width, 1024))
        decision_model = DecisionNet().network(mask_shape=input_1.shape,
                                               feature_shape=input_2.shape,
                                               num_class=self.num_class)
        output = decision_model([input_1, input_2])

        self.assertEqual((self.batch_size, self.num_class), np.shape(output))

        # TEST 3: feature map 의 가로, 세로 사이즈가 논문의 설명대로 1/8로 줄어드는지 확인
        height, width = self.official_input_size[1]
        input_1 = np.random.random_sample(size=(self.batch_size, height, width, 1))
        input_2 = np.random.random_sample(size=(self.batch_size, height, width, 1024))
        decision_model = DecisionNet().network(mask_shape=input_1.shape,
                                               feature_shape=input_2.shape,
                                               num_class=self.num_class)

        self.assertEqual((height//8, width//8), np.shape(decision_model.get_layer(name='dec_conv_3').output)[1:3])

    def test_build_model(self):
        # TEST 1: No Pretrained 인 경우에 정상 동작
        height, width = self.official_input_size[0]
        input_1 = np.random.random_sample(size=(self.batch_size, height, width, 1))
        input_2 = np.random.random_sample(size=(self.batch_size, height, width, 1024))
        decision_model = DecisionNet().build_model(pretrained=None,
                                                   mask_shape=input_1.shape,
                                                   feature_shape=input_2.shape,
                                                   num_class=self.num_class)
        output = decision_model([input_1, input_2])

        self.assertEqual((self.batch_size, self.num_class), np.shape(output))

        # TEST 2: No Pretrained 인데 network 생성에 필요한 config 가 없는 경우
        self.assertRaises(Exception,
                          DecisionNet().build_model,
                          pretrained='')

        # TEST 3: Pretrained Model Load
        # #텐서플로 라이브러리를 그대로 사용하므로 테스트 코드를 생략함.