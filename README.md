# Conditional handwriting generation

![img](https://raw.githubusercontent.com/adbrebs/handwriting/master/sous_le_pont_Mirabeau.png?token=AGnjokequVSx2LtbQW_UcGMmqoNg9kzHks5XGAtMwA%3D%3D "Guillaume Apollinaire")

Reproduces Alex Graves paper [Generating Sequences With Recurrent Neural Networks](http://arxiv.org/abs/1308.0850).  
Thanks to [Jose Sotelo](https://github.com/sotelo/) and [Kyle Kastner](https://github.com/kastnerkyle) for helpful discussions.

### Requirements

- Theano ('0.8.2')
- Lasagne (0.1) (just for the Adam optimizer)
- [Raccoon](https://github.com/adbrebs/raccoon): version 98b42b21e6df0ce09eaa7b9ad2dd10fcc993dd85  
    git submodule update --init --recursive  
- Apply patch to "/var/opt/wba/apps/anaconda2/lib/python2.7/site-packages/theano/gof/op.py"
$ patch -R op.py theano_gof_op.patch

### Generate the data
Download the following files and unpack them in a directory named 'handwriting':

- http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database/download-the-iam-on-line-handwriting-database: the files lineStrokes-all.tar.gz and ascii-all.tar.gz
- https://raw.githubusercontent.com/szcom/rnnlib/master/examples/online_prediction/validation_set.txt
- https://github.com/szcom/rnnlib/blob/master/examples/online_prediction/training_set.txt

Add an environment variable $DATA_PATH containing the parent directory of 'handwriting'.

### Train the model
Set an environment variable $TMP_PATH to a folder where the intermediate results, parameters, plots will be saved during training.

If you want to change the training configuration, modify the beginning of main_cond.py.
Run main_cond.py. By default, it is a one-layer GRU network with 400 hidden units.  
There is no ending condition, so you will have to stop training by doing ctrl+c.

It takes a few hours to train the model on a high-end GPU.

### Generate sequences
Run
python sample_model.py -f $TMP_PATH/handwriting/EXP_ID/f_sampling.pkl -s 'Sous le pont Mirabeau coule la Seine.' -b 0.7 -n 2
where EXP_ID is the generated id of the experiment you launched.

![img](https://raw.githubusercontent.com/adbrebs/handwriting/master/sous_le_pont_Mirabeau_2.png "Guillaume Apollinaire")


## DAN notes:
vi /var/opt/wba/apps/anaconda2/lib/python2.7/site-packages/lasagne/layers/pool.py
- I removed the include of downsample

- generate data : $ DATA_PATH=. python data_raw2hdf5.py  


