This is a simple unsupervised segmentation based method for turning a raster image into SVG.

![](https://raw.githubusercontent.com/mehdidc/pixel_to_svg/master/examples/flower_result.png)

# How to install?

You first need to install pypotrace.

Here are the steps to install pypotrace:

1. `sudo apt-get install build-essential python-dev libagg-dev libpotrace-dev pkg-config`
2. `git clone https://github.com/mehdidc/pypotrace`
3. `cd pypotrace`
4. `git checkout to_xml`
5. `rm -f potrace/*.c potrace/*.cpp potrace/agg/*.cpp potrace/*.so potrace/agg/*.so`
6. `pip install .`


Once pypotrace is available, you can install this repo.
Here are the steps: 

1. `git clone https://github.com/mehdidc/pixel_to_svg`
2. `cd pixel_to_svg`
3. `python setup.py install`


# How to use ?

Please check the example in <https://github.com/mehdidc/pixel_to_svg/tree/master/examples>




