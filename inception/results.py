import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


def save_confusion_matrix(sess, name, data, n_classes, y_pred_cls):
    # Get the true classifications for the test-set.
    cls_true = data.labels

    # Get the predicted classifications for the test-set.
    cls_pred = sess.run(y_pred_cls)

    # cls_true are one hot vectors, so convert to multiclass:
    cls_true_multiclass = []
    for image in cls_true:
        index = np.where(image == 1)
        cls_true_multiclass.append(index[0][0])

    # requre numpy array not list
    cls_true_multiclass = np.array(cls_true_multiclass)

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true_multiclass,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print("confusion matrix: \n", cm)

    # Plot the confusion matrix as an image.
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Make adjustments to the plot.
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, range(n_classes))
    plt.yticks(tick_marks, range(n_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(name + '_metrics.pdf', bbox_inches='tight', format='pdf')
    plt.show()

res = pd.read_csv('result.csv')
pass