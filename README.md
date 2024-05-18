# Advanced-CapsNet-Techniques-Incorporating-Custom-Squashing-Pretraining-and-Routing-For-BM-
Introduction: Capsule Networks (CapsNets) offer an exciting alternative to traditional CNNs for classification tasks. However, they face challenges concerning computational efficiency and information loss during integration with CNN layers. 


Our Approach: This repository introduces a novel approach that tackles these limitations. We propose replacing conventional CNN layers within CapsNet with pre-trained models like VGG-CapsNet and GN-CapsNet. This leverages the powerful feature extraction capabilities of these models within your CapsNet architecture. Furthermore, we've implemented modifications to dynamic structures and the squash function using custom functions. These enhance dynamic routing and feature modulation within the capsule layers, boosting their adaptability and flexibility. 


Evaluation: The method's effectiveness is evaluated on diverse datasets spanning various domains: Bone Marrow (BM) MNIST Fashion-MNIST CIFAR-10 These datasets showcase the approach's applicability in areas like medical imaging and object recognition. 


Results: Our experiments demonstrate significant improvements in classification performance compared to traditional CapsNet and CNN-based architectures. Additionally, the approach exhibits robustness against adversarial attacks, underlining its real-world viability. Performance Highlights: Bone Marrow (BM) Dataset VGG-CapsNet: 99.31% accuracy GN-CapsNet: 99.89% accuracy MNIST Dataset VGG-CapsNet: 99.87% accuracy GN-CapsNet: 99.98% accuracy Fashion-MNIST Dataset VGG-CapsNet: 95.22% accuracy GN-CapsNet: 95.95% accuracy CIFAR-10 Dataset VGG-CapsNet: 83.58% accuracy GN-CapsNet: 86.54% accuracy These results demonstrate superior performance on complex datasets, validating the effectiveness of our proposed approach.



 Conclusion: This method leverages the strengths of pre-trained models to significantly enhance CapsNet by addressing its computational inefficiencies and information loss issues. The improvements in classification accuracy and robustness against adversarial attacks highlight the potential of this approach for diverse real-world applications.
