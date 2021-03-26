def rnn_classification_visualize(questions_list_tokenized, labels_list):
    """
    Visualization method for the nlp model classes
    Args:
        questions_list_tokenized: a tokenized list corresponding to the input text
        labels_list: a list of predicted labels for each input text
    Return:
        display of the result stored in a string
    """
    result = "{:15} | {:5}\n".format("Word", "Pred")
    result += "=" * 20
    result += "\n"
    for i in range(len(labels_list)):
        for word, label in zip(questions_list_tokenized[i], labels_list[i]):
            label = label.replace("B-geo", "Geographical Entity")
            label = label.replace("I-geo", "Geographical Entity")
            label = label.replace("B-tim", "Time indicator")
            label = label.replace("I-tim", "Time indicator")
            label = label.replace("B-org", "Organization")
            label = label.replace("I-org", "Organization")
            label = label.replace("B-gpe", "Geopolitical Entity")
            label = label.replace("I-gpe", "Geopolitical Entity")
            label = label.replace("B-per", "Person")
            label = label.replace("I-per", "Person")
            label = label.replace("B-eve", "Event")
            label = label.replace("I-eve", "Event")
            label = label.replace("B-art", "Artifact")
            label = label.replace("I-art", "Artifact")
            label = label.replace("B-nat", "Natural Phenomenon")
            label = label.replace("I-nat", "Natural Phenomenon")
            label = label.replace("O", "no Label")
            result += "{:15} | {:5}\n".format(word, label)
        result += "\n"
    return result
