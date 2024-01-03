python tools/gen_neg_cls.py
python tools/test_robust.py configs/diseased/mobilenet_cancer.py work_dirs/mobilenet_cancer/latest.pth --metrics accuracy precision recall f1_score f2_score
python tools/test_robust.py configs/diseased/shuffle_cancer.py work_dirs/shuffle_cancer/latest.pth --show-dir work_dirs/shuffle_cancer/images