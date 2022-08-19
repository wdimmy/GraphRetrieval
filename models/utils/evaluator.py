class CMetrics:
    def __init__(self, metric_name):
        self.metric_name = metric_name


    def eval(self, input_dict):
        if self.metric_name == "acc":
             return self.accuracy_MNIST_CIFAR(input_dict)


    def accuracy_MNIST_CIFAR(self, input_dict):
        # acc_list = []
        # for i in range(y_true.shape[1]):
        #     is_labeled = y_true[:,i] == y_true[:,i]
        #     correct = y_true[is_labeled,i] == y_pred[is_labeled,i]
        #     acc_list.append(float(np.sum(correct))/len(correct))

        # return {'acc': sum(acc_list)/len(acc_list)}
        y_true, y_pred = input_dict["y_true"], input_dict["y_pred"]
        y_pred = y_pred.argmax(axis=1)
        acc = (y_pred == y_true).sum()
        return {"acc": acc/len(y_true)}