def return_string(tag, loss, accuracy):
    string = "{0} Loss: {1:.3f} Accuracy: {2:.3f}"
    return string.format(tag, loss, accuracy)