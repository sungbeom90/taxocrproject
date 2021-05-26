# 손실함수 - 기존 numpy.array 에서 tensor type으로 바꿈
def loss_region(y_true, y_pred):
    result = K.sum(K.square(y_true - y_pred), axis=-1)
    return result
