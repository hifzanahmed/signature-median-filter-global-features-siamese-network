import numpy as np
import cv2
from tslearn.metrics import dtw

class Utilities:
    @staticmethod
    def load_image(image_path):
        import cv2
        # Load the image
        signature = cv2.imread(image_path)
        
        # Check if image is loaded correctly
        if signature is None:
            print("Error: Signature Image not found or unable to load.")
            return None
        else:
            #print("Signature Image loaded successfully.")
            # Show the image
            cv2.imshow("Signature Image", signature)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return signature
        
    @staticmethod
    def crop_signature(binary_img):
        """
        Crops the binary image by removing all-zero rows and columns.
        Args:
        binary_img (np.ndarray): 2D numpy array where 1s represent signature, 0s are background.
        Returns:
        np.ndarray: Cropped binary image.
        """
        assert binary_img.ndim == 2, "Input must be a 2D array"
        # Find rows and columns with at least one '1'
        rows = np.any(binary_img, axis=1)
        cols = np.any(binary_img, axis=0)
        # Crop the image
        cropped_img = binary_img[rows][:, cols]
        return cropped_img
    
    @staticmethod
    def crop_and_resize_signature(binary_img):
        """
        Crops the binary image by removing all-zero rows and columns.
        Args:
        binary_img (np.ndarray): 2D numpy array where 1s represent signature, 0s are background.
        Returns:
        np.ndarray: Cropped binary image.
        """
        assert binary_img.ndim == 2, "Input must be a 2D array"
        # Find rows and columns with at least one '1'
        rows = np.any(binary_img, axis=1)
        cols = np.any(binary_img, axis=0)
        # Crop the image
        cropped_img = binary_img[rows][:, cols]
        return Utilities.resize_image(cropped_img)
    
    @staticmethod
    def resize_image(img, size=(300, 150)):
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def extract_features(cropped_img):
        processed_image_features = Utilities.horizontal_vertical_projection(cropped_img)
        #print("Horizontal and Vertical Projection:", processed_image_features.size)
        return processed_image_features
    
    @staticmethod
    def extract_features_discrete_radon_transform(cropped_img):
        processed_image_features = Utilities.horizontal_vertical_projection_discrete_radon_transform(cropped_img)
        print("Horizontal and Vertical Projection:", processed_image_features.size)
        return processed_image_features

    @staticmethod
    def horizontal_projection(binary_img):
        """
        Calculate horizontal projection of a binary image.
        Args:
        binary_img (np.ndarray): 2D binary image (0s and 1s)
        Returns:
        np.ndarray: 1D array containing sum of pixels in each row
        """
        assert binary_img.ndim == 2, "Image must be 2D"
        return np.sum(binary_img, axis=1)
    
    @staticmethod
    def vertical_projection(binary_img):
        """
        Calculate vertical projection of a binary image.
        Args:
        binary_img (np.ndarray): 2D binary image (0s and 1s)
        Returns:
        np.ndarray: 1D array containing sum of pixels in each row
        """
        assert binary_img.ndim == 2, "Image must be 2D"
        return np.sum(binary_img, axis=0)
    
    @staticmethod
    def horizontal_vertical_projection(binary_img):
        """
        Calculate horizontal and vertical projections of a binary image.
        Args:
        binary_img (np.ndarray): 2D binary image (0s and 1s)
        Returns:
        tuple: (horizontal_proj, vertical_proj)
            - horizontal_proj: 1D array of sums across rows
            - vertical_proj: 1D array of sums across columns
        """
        assert binary_img.ndim == 2, "Input must be 2D array"
        horizontal_proj = np.sum(binary_img, axis=1)
        vertical_proj = np.sum(binary_img, axis=0)
        return np.concatenate((horizontal_proj, vertical_proj))
    
    @staticmethod
    def discrete_radon_transform(binary_img):
        # 0° projection (horizontal features): sum along rows
        horizontal_projection = np.sum(binary_img, axis=1)
        # 90° projection (vertical features): sum along columns
        vertical_projection = np.sum(binary_img, axis=0)
        return horizontal_projection, vertical_projection
    
    @staticmethod
    def horizontal_vertical_projection_discrete_radon_transform(binary_img):
        horizontal_proj, vertical_proj = Utilities.discrete_radon_transform(binary_img)
        return np.concatenate((horizontal_proj, vertical_proj))
    
    @staticmethod
    def compute_training_score(signatures):
        """
        Computes S1: average DTW distance between all pairs of genuine signatures.
        Args:
        signatures (list): List of K signature samples (e.g., time series data).
        dtw_distance_func (callable): Function to compute DTW distance between two signatures.
        Returns:
        float: S1 training score.
        """
        K = len(signatures)
        total_dist = 0.0
        count = 0
        utils = Utilities()  
        print("signature list size:", K)
        for  i in range(K - 1):
            for j in range(i + 1, K):
                dist = utils.dtw(signatures[i], signatures[j])
                total_dist += dist
                count += 1

        if count == 0:
            return 0  # Avoid division by zero
        return (2 / (K * (K - 1))) * total_dist
    
    @staticmethod
    def compute_verification_score(test_signature, genuine_signatures):
        """
        Computes S2: average DTW distance between a test signature and all genuine signatures.
        Args:
        test_signature: The test signature sample (e.g., time series data).
        genuine_signatures (list): List of K genuine signature samples.
        dtw_distance_func (callable): Function to compute DTW distance between two signatures.
        Returns:
        float: S2 verification score.
        """
        K = len(genuine_signatures)
        total_dist = 0.0
        utils = Utilities()  
        for i in range(K):
            dist = utils.dtw(test_signature, genuine_signatures[i])
            total_dist += dist

        if K == 0:
            return 0  # Avoid division by zero
        return total_dist / K

    @staticmethod
    def dtw(signature1, signature2):
        # Logic to compute DTW (Dynamic Time Wraping) distance between two signatures
        return dtw(signature1, signature2)