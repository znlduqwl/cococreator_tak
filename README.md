# Show bounding boxes and labels
I added codes for showing the bounding boxes and labels. To show it, I used Jupyter Notebook and created deployer.ipynb and visualize_cg.ipynb files.

- Directory Structure  
```Shell
    |__ examples					
        |__ crestedgecko
        	|__ deployer.ipynb
        	|__ visualize_cg.ipynb
            	|__ train
            		|__ annotations
            			|__ CG_train2018_000000000001_patternless_0.png
            			|__ instances_crestedgecko_train2018
            		|__ crestedgecko_train2018
            			|__ CG_train2018_000000000001.jpg
        |__ shapes
        	|__ ...	        
```
- [deployer.ipynb](https://github.com/asyncbridge/pycococreator/blob/master/examples/crestedgecko/deployer.ipynb)  
1) Rename and resize image files.  
2) Create an annotation file from images and masking images.    
- [visualize_cg.ipynb](https://github.com/asyncbridge/pycococreator/blob/master/examples/crestedgecko/visualize_cg.ipynb)  
  
Show bounding boxes and labels including segmentations from the created annotation file.  
![alt text](https://github.com/asyncbridge/pycococreator/blob/master/examples/crestedgecko/output_6_0.png "output")

# Install

1. `pip install git+git://github.com/waspinator/pycococreator.git@0.2.0`  
2. Run Jupyter Notebook.(Set working directory as pycococreator)  
3. Go to pycococreator/examples/crestedgecko directory.  
4. Open deployer.ipynb and run it.  
- instances_crestedgecko_train2018.json annotation file will be created in pycococreator/examples/crestedgecko/train directory.
5. Copy the annotation file to pycococreator/examples/crestedgecko/annotations directory.
6. Open visualize_cg.ipynb and run it to show the sample image.  
- Segmentation, bounding box and label will be drawn on the sample image.  

# License
This project is made available under the [Apache 2.0 License](https://github.com/asyncbridge/pycococreator/blob/master/LICENSE).  
  
It is forked from: [https://github.com/waspinator/pycococreator](https://github.com/waspinator/pycococreator)  

# Reference
[1] [https://patrickwasp.com/create-your-own-coco-style-dataset](https://patrickwasp.com/create-your-own-coco-style-dataset)