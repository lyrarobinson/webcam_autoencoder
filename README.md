# webcam_autoencoder
 Generate images from a live webcam feed using an autoencoder model<br>
 These are some scripts from part of an installation using a grid of light dependent resistors and machine learning
 to create a sort of camera - which is why all the images have to be squares


clone the repo, cd into that directory:
```
gh repo clone lyrarobinson/clipcat_tool
cd webcam-autoencoder
```

recommended to make a new environment with conda:
```
conda create -n webcam_autoencoder
conda activate webcam_autoencoder
```

install the packages from the setup.py file:
```
pip install -e .
```

then it can be used by typing this into the terminal:
```
webcam_autoencoder
```

You can also make your own autoencoder by gathering a dataset of 224x224 images, using the makegrids.py file on it, and then following the createmodel notebook. <br>
After a model is created, it is much easier to carry on training it using the train notebook


