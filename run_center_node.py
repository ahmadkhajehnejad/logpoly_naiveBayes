
from classifier.center_model import NaiveBayesClassifier, scorer
from communication_tools import load_data
from config.client_nodes_address import client_nodes_address
from communication_tools import send_msg


if __name__ == '__main__':

    features_info, _, _, data_test, labels_test = load_data()
    classifier = NaiveBayesClassifier(features_info)
    classifier.fit()

    send_msg(client_nodes_address, ['close'], 0)

    # score = scorer(classifier, data_test, labels_test)
    # print('\ntest score:  ', score)


