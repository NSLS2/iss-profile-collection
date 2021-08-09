#!/bin/bash

export AZURE_TESTING=1

pip install -vv git+https://github.com/NSLS-II-ISS/isstools@main
pip install -vv git+https://github.com/NSLS-II-ISS/xas@master
pip install -vv git+https://github.com/NSLS-II-ISS/xview@master
pip install -vv git+https://github.com/NSLS-II-ISS/isscloudtools@master "oauth2client<4.0.0"


# Create non-standard directories:
sudo mkdir -v -p /nsls2/xf08id/metadata/

sudo chmod -Rv go+rw /nsls2/xf08id/

# touch /nsls2/xf08id/...

