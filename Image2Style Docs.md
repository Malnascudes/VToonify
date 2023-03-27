To store the encodings of an image as a new style run the following script

```
python image_2_style.py --cpu \
--style_image_path '/Users/carlesanton/repos/matriu/ideal/Datasets/styles/elf/elf-disney.jpeg' \
--exstyle_path './checkpoint/vtoonify_d_cartoon/exstyle_code_custom.npy' \
--style_encoder_path './checkpoint/encoder.pt' \
--ckpt './checkpoint/vtoonify_d_cartoon/vtoonify_s_d_c.pt'
```

The script saves two style vectors:
* no-zplus2wplus-<image-name>: straight result of pspencoder(). Saved at postion -1
* <image-name>: after applying vtoonify.zplus2wplus(s_w). Saved at postion -2

Results are better using the `no-zplus2wplus-<image-name>` style, color transfer and eyes colors are changed.
Test encoding with [this image](https://lumiere-a.akamaihd.net/v1/images/open-uri20150422-20810-1mtgk81_b7c97ea6.jpeg?region=0,0,450,450)


To use the saved encoding run. Replace the `content` argument path for the desired image.
Make sure that `exstyle_path`, `style_encoder_path` and `ckpt` are the same used in the `image_2_style.py` experiment.

```
python style_transfer.py --scale_image --cpu \
--content '/Users/carlesanton/repos/matriu/ideal/Datasets/matriu-faces-white-bg/dani4.png' \
--exstyle_path './checkpoint/vtoonify_d_cartoon/exstyle_code_custom.npy' \
--style_encoder_path './checkpoint/encoder.pt' \
--ckpt './checkpoint/vtoonify_d_cartoon/vtoonify_s_d_c.pt' \
--output_path './outputs/upload_test_disney_elf/' \
--style_id -1 \
--style_degree 0.4 --color_transfer
```

`--style_id -1` accesses the last position of the styles. Used style name can be checked in the logs.

