Implementation of [*KaiMing He el.al. Masked Autoencoders Are Scalable Vision Learners*](https://arxiv.org/abs/2111.06377).

Due to limit resource available, we only test the model on cifar10. We mainly want to reproduce the result that **pre-training an ViT with MAE can achieve a better result than directly trained in supervised learning with labels**. This should be an evidence of **self-supervised learning is more data efficient than supervised learning**.

|Model|Test Acc|
|-----|--------|
|ViT-T||
|ViT-T-MAE||