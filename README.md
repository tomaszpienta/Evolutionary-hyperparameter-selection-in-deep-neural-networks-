## **Evolutionary hyperparameter selection**

The project is based on the selection of evolutionary hyperparameters in deep neural networks.
 - Database is a three-dimensional hyperspectral image model (PaviaU, KSC, Botswana)
 - Hyperparameters trained using Genetic Algorithm
 - Model is written using the PyTorch library


### Project Structure
```
├──Dataloaders
|    ├── Pavia_dataloader.py
|    ├── KSC_dataloader.py
|    ├── indian_pines_dataloader.py    
│    └── Botswana_dataloader.py    
├──Loadset
|    ├──PaviaU
|    |   ├── data
│    |       ├── PaviaU.mat    
│    |       └── PaviaU_gt.mat
|    ├──KSC
|    |   ├── data
│    |       ├── KSC.mat    
│    |       └── KSC_gt.mat
|    ├──Botswana
|    |   ├── data
│    |       ├── Botswana.mat    
│    |       └── Botswana_gt.mat
├── Wyniki_IndianPines
├── Wyniki_KSC
├── Wyniki_PaviaU
├── main.py           
├── models.py         
├── Hyperspectra_grids.py


```

### CNN model preview

```python
class C3D(nn.Module):
    def __init__(self, output, _stride, kernel_size, D_out, H_out, W_out, output2, kernel_size2, _stride2):
        super(C3D, self).__init__()     
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, output, kernel_size, stride=_stride),
            nn.ReLU(),
            nn.Conv3d(output, output2, kernel_size2, stride=_stride2),
            nn.Softmax(dim=1))
        self.fc = nn.Linear(D_out*H_out*output2*W_out, D_out*H_out)   
        
    def forward(self, x, D_out, H_out):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = out.reshape(D_out, H_out)

        return out 
```
