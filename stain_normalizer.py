

def StainNormalizer(object):

    def __init__(self, method):
        self.method = method.lower()

    def fit(self, I):
        """
        Set target image

        :param I:
        :return:
        """
        pass

    def transform(self, I):
        """
        Get color-corrected image

        :param I:
        :return:
        """
        pass