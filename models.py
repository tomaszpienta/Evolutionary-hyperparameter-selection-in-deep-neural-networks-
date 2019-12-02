'''DATALOADERS'''

def LoadModel(name):
#    print(name)
#    classes = []
#    D = 0
#    H = 0
#    W = 0
#    Height = 0
#    Width = 0
#    Bands = 0
#    samples = 0

    if name == 'PaviaU':   
        
        classes = []
        #Size of 3D images        
        D = 610
        H = 340
        W = 103
        #Size of patches        
        Height = 27
        Width = 21
        Bands = 103
        samples = 2400
        patch = 300000
    
    elif name == 'IndianPines':

        classes = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                   "Corn", "Grass-pasture", "Grass-trees",
                   "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                   "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                   "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                   "Stone-Steel-Towers"]
        #Size of 3D images        
        D = 145
        H = 145
        W = 200
        #Size of patches        
#        Height = 17 
#        Width = 17
        Height = 19
        Width = 17
        Bands = 200
        samples = 4000
        patch = 300000
        
    elif name == 'Botswana':

        classes = ["Undefined", "Water", "Hippo grass",
                    "Floodplain grasses 1", "Floodplain grasses 2",
                    "Reeds", "Riparian", "Firescar", "Island interior",
                    "Acacia woodlands", "Acacia shrublands",
                    "Acacia grasslands", "Short mopane", "Mixed mopane",
                    "Exposed soils"]
        
        #Size of 3D images
        D = 1476 
        H = 256
        W = 145
        #Size of patches
        Height = 31
        Width = 21
        Bands = 145
        samples = 1500
        patch = 30000
        
    elif name == 'KSC':

        classes = ["Undefined", "Scrub", "Willow swamp",
                    "Cabbage palm hammock", "Cabbage palm/oak hammock",
                    "Slash pine", "Oak/broadleaf hammock",
                    "Hardwood swamp", "Graminoid marsh", "Spartina marsh",
                    "Cattail marsh", "Salt marsh", "Mud flats", "Wate"]
        #Size of 3D images
        D = 512
        H = 614
        W = 176
        #Size of patches        
        Height = 31
        Width = 27
        Bands = 176
        samples = 2400
        patch = 30000
        
        
    elif name == 'Salinas':

        classes = []
        #Size of 3D images
        D = 512
        H = 217
        W = 204
        #Size of patches        
        Height = 21
        Width = 17
        Bands = 204
        samples = 2600
        patch = 300000
        
    elif name == 'SalinasA':

        classes = []
        #Size of 3D images
        D = 83
        H = 86
        W = 204
        #Size of patches        
        Height = 15
        Width = 11
        Bands = 204
        samples = 2400
        patch = 300000
        
    elif name == 'Samson':

        classes = []
        #Size of 3D images
        D = 95
        H = 95
        W = 156
        #Size of patches        
        Height = 10
        Width = 10
        Bands = 156
        samples = 1200
        patch = 30000
        
    return Width, Height, Bands, samples, D, H, W, classes, patch    
