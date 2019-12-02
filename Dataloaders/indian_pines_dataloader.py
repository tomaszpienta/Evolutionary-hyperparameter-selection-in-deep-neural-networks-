import numpy as np
from scipy import io
import os

g_nr_of_clusters = 17

def numpy_to_uint8(the_image, verbal=False):
    original_image_type = the_image.dtype
    original_image_info = np.iinfo(original_image_type)
    converted_image = the_image.astype(np.float32) / original_image_info.max
    converted_image_type = np.uint8
    converted_image_info = np.iinfo(converted_image_type)
    converted_image = converted_image_info.max * converted_image
    converted_image = converted_image.astype(np.uint8)
    if verbal:
        print("Original image type: ", original_image_type)
        print("Converted image type:  ", converted_image_type)
    return converted_image

class Dataloader:
    
    def __init__(self):
    # self.data_dir = 'C:/Users/Public/AI/artificial-intelligence---my-beginning/venv/data/Pavia/'
        self.data_dir = './Loadset/IndianPines/data/'
        if not os.path.exists(self.data_dir):
            self.data_dir = '/IndianPines/data/'


        self.image_shape = (145 , 145, 200)
        self.image = ()
   
    
    def get_image(self):  
            
        filename = 'Indian_pines_corrected.mat'
        ImDict = io.loadmat(self.data_dir + filename)
        image_name = 'indian_pines_corrected'
        the_image = ImDict[image_name]
        image_size = np.shape(the_image)
        NRows = image_size[0]
        NCols = image_size[1]
        depth = image_size[2]
        print("Lokalizacja obrazu: \t", self.data_dir + filename)
        print("Nazwa obrazu:  \t\t\t", image_name)
        print("Rozmiar: \t\t\t\t", "wiersze: ", NRows, " kolumny: ", NCols, " głębokość: ", depth)
        print("Ilośc pikseli (ilość kolumn * ilość wierszy): ", NRows * NCols)

        print()
        print("***   Converting image to uint8   ***")
        print("---------------------------------")
        the_image = numpy_to_uint8(the_image)

        self.image = the_image
        

        return self.image
        
        
    def get_labels(self, verbal=True):
        if verbal:
            print()
            print("***   Loading labels   ***")
            print("---------------------------------")
    
    
            # To juz jest w uint8
        filename_labels = 'Indian_pines_gt.mat'
        ImDict_labels = io.loadmat(self.data_dir + filename_labels)
        image_name_labels = 'indian_pines_gt'
        the_image_labels = ImDict_labels[image_name_labels]
    
        # labels unification - wartości od 0 do number_of_labels -1
        unused_label = 0
        labels_dictionary = {}
        x = 0
        y = 0
        labels_values = set()
        for i in range(self.image_shape[0] * self.image_shape[1]):
            if the_image_labels[y, x] not in labels_dictionary:
                labels_dictionary[the_image_labels[y, x]] = unused_label
                unused_label += 1
            the_image_labels[y, x] = labels_dictionary[the_image_labels[y, x]]
            labels_values.add(the_image_labels[y, x])
            x = x + 1
            if x == self.image_shape[1]:
                x = 0
                y += 1
    
        image_size_labels = np.shape(the_image_labels)
        NRows_labels = image_size_labels[0]
        NCols_labels = image_size_labels[1]
    
        # import matplotlib.pyplot as plt
        # plt.imshow(the_image_labels)
        # plt.show()
    
        if verbal:
            print("Lokalizacja obrazu: \t", filename_labels)
            print("Nazwa obrazu:  \t\t\t", image_name_labels)
            print("Rozmiar: \t\t\t\t", "wiersze: ", NRows_labels, " kolumny: ", NCols_labels)
            #print("Ilośc etykiet: \t\t\t", self.nr_of_clusters)
            print("Etykiety: \t\t\t\t", labels_values)
        self.image_labels = the_image_labels
    
        return self.image_labels