To store the encodings of an image as a new style run the following script

```
python image_2_style.py --cpu \
--style_image_path '/Users/carlesanton/repos/matriu/ideal/Datasets/styles/elf/elf-disney.jpeg' \
--exstyle_path './checkpoint/vtoonify_d_cartoon/exstyle_code_custom.npy' \
--style_encoder_path './checkpoint/encoder.pt' \
--ckpt './checkpoint/vtoonify_d_cartoon/vtoonify_s_d_c.pt'
```


To use the saved encoding run

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

