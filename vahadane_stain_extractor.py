import spams
import cv2 as cv
import numpy as np

class VahadaneStainExtractor(object):

    def __init__(self, I):
        self.image = I

    def get_stain_matrix(self, I, luminosity_threshold=0.8, regularizer=0.1):
        """
        Stain matrix estimation via method of:
        A. Vahadane et al. 'Structure-Preserving Color Normalization and Sparse Stain Separation for Histological Images'
        :param I: Image RGB uint8.
        :param luminosity_threshold:
        :param regularizer:
        :return:
        """
        # convert to OD and ignore background
        tissue_mask = self.get_tissue_mask(I, luminosity_threshold=luminosity_threshold).reshape((-1,))
        OD = self.convert_RGB_to_OD(I)
        OD = OD.reshape((-1, 3))
        OD = OD[tissue_mask]

        # do the dictionary learning
        dictionary = spams.trainDL(X=OD.T, K=2, lambda1=regularizer, mode=2,
                                   modeD=0, posAlpha=True, posD=True, verbose=False).T

        # order H and E.
        # H on first row.
        if dictionary[0, 0] < dictionary[1, 0]:
            dictionary = dictionary[[1, 0], :]

        return self.normalize_matrix_rows(dictionary)

    def get_concentrations(self, I, stain_matrix, regularizer=0.01):
        """
        Estimate concentration matrix given an image and stain matrix.
        :param I:
        :param stain_matrix:
        :param regularizer:
        :return:
        """
        OD = self.convert_RGB_to_OD(I).reshape((-1, 3))
        return spams.lasso(X=OD.T, D=stain_matrix.T, mode=2, lambda1=regularizer, pos=True).toarray().T


    def get_tissue_mask(self, I, luminosity_threshold=0.8):
        """
        Get a binary mask where true denotes pixels with a luminosity less than the specified threshold.
        Typically we use to identify tissue in the image and exclude the bright white background.
        
        :param I: RGB uint 8 image.
        :param luminosity_threshold: Luminosity threshold.
        :return: Binary mask.
        """
        I_LAB = cv.cvtColor(I, cv.COLOR_RGB2LAB)
        L = I_LAB[:, :, 0] / 255.0  # Convert to range [0,1].
        mask = L < luminosity_threshold

        # Check it's not empty
        # if mask.sum() == 0:
        #     raise TissueMaskException("Empty tissue mask computed")

        return mask

    def convert_RGB_to_OD(self, I):
        """
        Convert from RGB to optical density (OD_RGB) space.
        RGB = 255 * exp(-1*OD_RGB).
        :param I: Image RGB uint8.
        :return: Optical denisty RGB image.
        """
        mask = (I == 0)
        I[mask] = 1
        return np.maximum(-1 * np.log(I / 255), 1e-6)

    def convert_OD_to_RGB(self, OD):
        """
        Convert from optical density (OD_RGB) to RGB.
        RGB = 255 * exp(-1*OD_RGB)
        :param OD: Optical denisty RGB image.
        :return: Image RGB uint8.
        """
        assert OD.min() >= 0, "Negative optical density."
        OD = np.maximum(OD, 1e-6)
        return (255 * np.exp(-1 * OD)).astype(np.uint8)

    def normalize_matrix_rows(self, A):
        """
        Normalize the rows of an array.
        :param A: An array.
        :return: Array with rows normalized.
        """
        return A / np.linalg.norm(A, axis=1)[:, None]