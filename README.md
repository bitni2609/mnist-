 #This program aims to recognize the handwriting picture 
 through the MNIST data set, a classic dataset for the data mining,and softmax algorithm, an useful function for machine learning. 

the dockerfile is for construct the docker iamge

requirement is sum of basic module of real1.py for download in docker iamges

xunlian.py is code for training mnist data set and get the file model.ckpt

model.ckpt is one model we can apply , which is a basic model for handwriting picture to be recognized

shibie.py is code for us to use the model.ckpt to realize the recognization of handwriting picture in local machine,
you can see the basic address of file located in my computer

real2.py is code for realizd  the recognization of handwriting picture in local server. we can upload file through local server rather than tell 
python the direct address. I made this through flask

real1.py is the final code for docker. we change the local file dir into the workdir in docker, and through docker, we can do the samething 
as we did through real2.py


the final report is the report1.pdf.
the whold process is stored in it!
