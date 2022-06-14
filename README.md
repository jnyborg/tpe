# Thermal Positional Encoding
Source code for the paper ["Generalized Classification of Satellite Image Time Series with Thermal Positional Encoding"](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/html/Nyborg_Generalized_Classification_of_Satellite_Image_Time_Series_With_Thermal_Positional_CVPRW_2022_paper.html) by Joachim Nyborg, Charlotte Pelletier, and Ira Assent.


## Requirements
- PyTorch 1.10.0
- Python 3.8.12
- Numpy 1.21.2

The TimeMatch dataset and our extension with weather data can be downloaded from [Zenodo](https://zenodo.org/record/6542639). The data classes and splits used for the paper can be found in the `dataset_extensions` directory.

## Usage
See `scripts/run_experiments.sh` for examples for how to train both calendar time and thermal time model variants.

## Citation
If you find the paper and/or the code useful for your work, please consider citing our paper:
```
@InProceedings{Nyborg_2022_CVPR,
    author    = {Nyborg, Joachim and Pelletier, Charlotte and Assent, Ira},
    title     = {Generalized Classification of Satellite Image Time Series With Thermal Positional Encoding},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {1392-1402}
}
```

## Credits
- The code builds upon the original [TimeMatch code](https://github.com/jnyborg/timematch)
- The implementation of PSE+LTAE is based on [the official implementation](https://github.com/VSainteuf/lightweight-temporal-attention-pytorch)
